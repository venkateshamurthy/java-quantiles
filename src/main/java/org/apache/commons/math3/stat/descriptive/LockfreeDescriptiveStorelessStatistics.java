/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.commons.math3.stat.descriptive;

import java.io.Serializable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import org.apache.commons.math3.exception.MathIllegalArgumentException;
import org.apache.commons.math3.exception.MathIllegalStateException;
import org.apache.commons.math3.exception.NullArgumentException;
import org.apache.commons.math3.stat.descriptive.moment.GeometricMean;
import org.apache.commons.math3.stat.descriptive.moment.Kurtosis;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.stat.descriptive.moment.Skewness;
import org.apache.commons.math3.stat.descriptive.moment.Variance;
import org.apache.commons.math3.stat.descriptive.rank.Max;
import org.apache.commons.math3.stat.descriptive.rank.Min;
import org.apache.commons.math3.stat.descriptive.rank.PSquarePercentile;
import org.apache.commons.math3.stat.descriptive.summary.Sum;
import org.apache.commons.math3.stat.descriptive.summary.SumOfSquares;
import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.util.MathArrays;
import org.apache.commons.math3.util.MathUtils;

import com.lmax.disruptor.EventFactory;
import com.lmax.disruptor.EventHandler;
import com.lmax.disruptor.EventTranslator;
import com.lmax.disruptor.EventTranslatorOneArg;
import com.lmax.disruptor.RingBuffer;
import com.lmax.disruptor.dsl.Disruptor;
import com.lmax.disruptor.util.DaemonThreadFactory;

/**
 * A Lockless/lockfree implementation of {@link DescriptiveStatisticalSummary
 * Statistical Summary} for a {@link StorelessUnivariateStatistic} using the
 * {@link Disruptor LMAX Disruptor concurrency utility}.
 */
public class LockfreeDescriptiveStorelessStatistics implements
		DescriptiveStatisticalSummary<StorelessUnivariateStatistic>,
		Serializable {

	/** Serialization UID */
	private static final long serialVersionUID = 4133067267405273064L;

	/** Default value for the {@code Disruptor ring buffer size}. */
	private static final int DEFAULT_WINDOW = 1024*4;

	/** {@code Disruptor} specifics */
	private transient final Disruptor<DoubleValue> disruptor;
	/** a double event factory required to publish events to a disruptor */
	private transient final EventFactory<DoubleValue> disruptorDataEventFactory = new DoubleValue.FACTORY();
	/** an event translator adhering to {2link {@link EventTranslator}*/
	private transient final DoubleValueProducerWithTranslator disruptorDataSourcer;
	/** thread pool*/
	private transient final ExecutorService disruptorExecutor = Executors
			.newCachedThreadPool(DaemonThreadFactory.INSTANCE);

	/** hold the current window size **/
	private final int windowSize;

	/** {@code StorelessUnivariateStatistic Statistic implementations} */
	/** Mean */
	private final StorelessUnivariateStatistic meanImpl;
	/** Geometric Mean */
	private final StorelessUnivariateStatistic geometricMeanImpl;
	/** Kurtosis */
	private final StorelessUnivariateStatistic kurtosisImpl;
	/** Max */
	private final StorelessUnivariateStatistic maxImpl;
	/** Min */
	private final StorelessUnivariateStatistic minImpl;
	/** Percentile */	
	private final StorelessUnivariateStatistic percentileImpl;
	/** Sum */
	private final StorelessUnivariateStatistic sumImpl;
	/** Sum Of Squares */
	private final StorelessUnivariateStatistic sumsqImpl;
	/** Skewness */
	private final StorelessUnivariateStatistic skewnessImpl;
	/** Variance */
	private final StorelessUnivariateStatistic varianceImpl;
	/** Array of Statistics */
	private final StorelessUnivariateStatistic[] storelessStats;

	/**
	 * An array of
	 * {@code StorelessUnivariateStatistic Statistic implementations} for
	 * convenience in looping
	 */
	private transient final StatisticEventHandler<? extends StorelessUnivariateStatistic>[] storelessStatEventHandlers;

	/**
	 * Default constructor that builds a {@link LockfreeDescriptiveStorelessStatistics} instance with
	 * a default size
	 */
	public LockfreeDescriptiveStorelessStatistics() {
		this(DEFAULT_WINDOW);
	}

	/**
	 * Construct a {@link LockfreeDescriptiveStorelessStatistics} instance with
	 * the specified window size
	 * 
	 * @param window size for the {@link RingBuffer ring}
	 */
	public LockfreeDescriptiveStorelessStatistics(int window)
			throws MathIllegalArgumentException {
		this(window, new Mean(), new GeometricMean(), new Kurtosis(),
				new Max(), new Min(), new PSquarePercentile(50d),
				new Skewness(), new Variance(), new SumOfSquares(), new Sum());
	}

	/**
	 * Construct a {@link LockfreeDescriptiveStorelessStatistics} instance with
	 * an initial data values in double[] initialDoubleArray. If
	 * initialDoubleArray is null, then this constructor corresponds to default
	 * constructor
	 * 
	 * @param initialDoubleArray  the initial double[].
	 */
	public LockfreeDescriptiveStorelessStatistics(double[] initialDoubleArray) {
		this();
		if (initialDoubleArray != null) {
			for (double d : initialDoubleArray){
				addValue(d);
			}
		}
	}

	/**
	 * Default constructor with typical set of statistics that implement a {@link StorelessUnivariateStatistic}
	 * 
	 * @param window size of RingBuffer
	 * @param mean is typically a {@link Mean} 
	 * @param geometricMean is typically a {@link GeometricMean} 
	 * @param kurtosis is typically a {@link Kurtosis} 
	 * @param max is typically a {@link Max} 
	 * @param min is typically a {@link Min} 
	 * @param percentile is typically a {@link PSquarePercentile} 
	 * @param skewness is typically a {@link Skewness} 
	 * @param variance is typically a {@link Variance}
	 * @param sumsq is  typically a {@link SumOfSquares}
	 * @param sum is typically a {@link Sum}
	 * @throws MathIllegalArgumentException in case if any of them are null
	 */
	private LockfreeDescriptiveStorelessStatistics(int window,
			StorelessUnivariateStatistic mean,
			StorelessUnivariateStatistic geometricMean,
			StorelessUnivariateStatistic kurtosis,
			StorelessUnivariateStatistic max,
			StorelessUnivariateStatistic min,
			StorelessUnivariateStatistic percentile,
			StorelessUnivariateStatistic skewness,
			StorelessUnivariateStatistic variance,
			StorelessUnivariateStatistic sumsq,
			StorelessUnivariateStatistic sum)
			throws MathIllegalArgumentException {
		
		storelessStats = new StorelessUnivariateStatistic[] {
				this.meanImpl = mean,
				this.geometricMeanImpl = geometricMean,
				this.kurtosisImpl = kurtosis, 
				this.maxImpl = max,
				this.minImpl = min, 
				this.percentileImpl = percentile,
				this.skewnessImpl = skewness,
				this.varianceImpl = variance, 
				this.sumsqImpl = sumsq,
				this.sumImpl = sum };
		checkNotNull((Object[])storelessStats);
		MathArrays.checkPositive(new double[]{window});
		this.windowSize = window;
		
		storelessStatEventHandlers = StatisticEventHandler.create(storelessStats);
		
		disruptor = new Disruptor<DoubleValue>(disruptorDataEventFactory,windowSize, disruptorExecutor);
		disruptor.handleEventsWith(storelessStatEventHandlers);
		
		final RingBuffer<DoubleValue> ring = disruptor.start();
		disruptorDataSourcer = new DoubleValueProducerWithTranslator(ring);
	}

	/******** With Functions ***************************/
	/**{@inheritDoc}*/
	public DescriptiveStatisticalSummary<StorelessUnivariateStatistic> withPercentile(
			StorelessUnivariateStatistic percentileImpl) {
		return new LockfreeDescriptiveStorelessStatistics(windowSize, meanImpl,
				geometricMeanImpl, kurtosisImpl, maxImpl, minImpl,
				percentileImpl, skewnessImpl, varianceImpl, sumsqImpl, sumImpl);
	}
	/**{@inheritDoc}*/
	public DescriptiveStatisticalSummary<StorelessUnivariateStatistic> withMean(
			StorelessUnivariateStatistic meanImpl) {
		return new LockfreeDescriptiveStorelessStatistics(windowSize, meanImpl,
				geometricMeanImpl, kurtosisImpl, maxImpl, minImpl,
				percentileImpl, skewnessImpl, varianceImpl, sumsqImpl, sumImpl);
	}
	/**{@inheritDoc}*/
	public DescriptiveStatisticalSummary<StorelessUnivariateStatistic> withKurtosis(
			StorelessUnivariateStatistic kurtosisImpl) {
		return new LockfreeDescriptiveStorelessStatistics(windowSize, meanImpl,
				geometricMeanImpl, kurtosisImpl, maxImpl, minImpl,
				percentileImpl, skewnessImpl, varianceImpl, sumsqImpl, sumImpl);
	}
	/**{@inheritDoc}*/
	public DescriptiveStatisticalSummary<StorelessUnivariateStatistic> withVariance(
			StorelessUnivariateStatistic varianceImpl) {
		return new LockfreeDescriptiveStorelessStatistics(windowSize, meanImpl,
				geometricMeanImpl, kurtosisImpl, maxImpl, minImpl,
				percentileImpl, skewnessImpl, varianceImpl, sumsqImpl, sumImpl);
	}
	/**{@inheritDoc}*/
	public DescriptiveStatisticalSummary<StorelessUnivariateStatistic> withSkewness(
			StorelessUnivariateStatistic skewnessImpl) {
		return new LockfreeDescriptiveStorelessStatistics(windowSize, meanImpl,
				geometricMeanImpl, kurtosisImpl, maxImpl, minImpl,
				percentileImpl, skewnessImpl, varianceImpl, sumsqImpl, sumImpl);
	}

	/**
	 * Adds the value to the dataset. If the dataset is at the maximum size
	 * (i.e., the number of stored elements equals the currently configured
	 * windowSize), the first (oldest) element in the dataset is discarded to
	 * make room for the new value.
	 * 
	 * @param value to be added
	 */
	public void addValue(double value) {
		disruptorDataSourcer.onData(value);

	}

	/**
	 * Returns the <a href="http://www.xycoon.com/arithmetic_mean.htm">
	 * arithmetic mean </a> of the available values
	 * 
	 * @return The mean or Double.NaN if no values have been added.
	 */
	public double getMean() {
		return meanImpl.getResult();
	}

	/**
	 * Returns the <a href="http://www.xycoon.com/geometric_mean.htm"> geometric
	 * mean </a> of the available values
	 * 
	 * @return The geometricMean, Double.NaN if no values have been added, or if
	 *         the product of the available values is less than or equal to 0.
	 */
	public double getGeometricMean() {
		return geometricMeanImpl.getResult();
	}

	/**
	 * Returns the (sample) variance of the available values.
	 * 
	 * 
	 * @return The variance, Double.NaN if no values have been added or 0.0 for
	 *         a single value set.
	 */
	public double getVariance() {
		return varianceImpl.getResult();
	}

	/**
	 * Returns the standard deviation of the available values.
	 * 
	 * @return The standard deviation, Double.NaN if no values have been added
	 *         or 0.0 for a single value set.
	 */
	public double getStandardDeviation() {
		double stdDev = Double.NaN;
		if (getN() > 0) {
			if (getN() > 1) {
				stdDev = FastMath.sqrt(getVariance());
			} else {
				stdDev = 0.0;
			}
		}
		return stdDev;
	}

	/**
	 * Returns the skewness of the available values. Skewness is a measure of
	 * the asymmetry of a given distribution.
	 * 
	 * @return The skewness, Double.NaN if no values have been added or 0.0 for
	 *         a value set &lt;=2.
	 */
	public double getSkewness() {
		return skewnessImpl.getResult();
	}

	/**
	 * Returns the Kurtosis of the available values. Kurtosis is a measure of
	 * the "peakedness" of a distribution
	 * 
	 * @return The kurtosis, Double.NaN if no values have been added, or 0.0 for
	 *         a value set &lt;=3.
	 */
	public double getKurtosis() {
		return kurtosisImpl.getResult();
	}

	/**
	 * Returns the maximum of the available values
	 * 
	 * @return The max or Double.NaN if no values have been added.
	 */
	public double getMax() {
		return maxImpl.getResult();
	}

	/**
	 * Returns the minimum of the available values
	 * 
	 * @return The min or Double.NaN if no values have been added.
	 */
	public double getMin() {
		return minImpl.getResult();
	}

	/**
	 * Returns the number of available values
	 * 
	 * @return The number of available values
	 */
	public long getN() {
		return disruptor.getRingBuffer().getCursor()+1;
	}

	/**
	 * Returns the sum of the values that have been added to Univariate.
	 * 
	 * @return The sum or Double.NaN if no values have been added
	 */
	public double getSum() {
		return sumImpl.getResult();
	}

	/**
	 * Returns the sum of the squares of the available values.
	 * 
	 * @return The sum of the squares or Double.NaN if no values have been
	 *         added.
	 */
	public double getSumsq() {
		return sumsqImpl.getResult();
	}

	/**
	 * Resets all statistics and storage
	 */
	public void clear() {
		disruptor.halt();
		disruptor.getRingBuffer().resetTo(RingBuffer.INITIAL_CURSOR_VALUE);
		for (StorelessUnivariateStatistic s : storelessStats){
			s.clear();
		}
	}
	
	public void halt(){
		disruptor.shutdown();
		disruptorExecutor.shutdownNow();
		try {
			disruptorExecutor.awaitTermination(-1, TimeUnit.MILLISECONDS);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}

	/**
	 * Returns the window of buffer used to intermediately store.
	 * 
	 * @return The current window size.
	 */
	public int getWindowSize() {
		return windowSize;
	}
	/** {@inheritDoc} */
	public double getPercentile() throws MathIllegalStateException,
			MathIllegalArgumentException {
		return percentileImpl.getResult();
	}

	/**
	 * Generates a text report displaying univariate statistics from values that
	 * have been added. Each statistic is displayed on a separate line.
	 * 
	 * @return String with line feeds displaying statistics
	 */
	@Override
	public String toString() {
		StringBuilder outBuffer = new StringBuilder();
		String endl = "\n";
		outBuffer.append(getClass().getSimpleName()).append(":").append(endl);
		outBuffer.append("n: ").append(getN()).append(endl);
		outBuffer.append("min: ").append(getMin()).append(endl);
		outBuffer.append("max: ").append(getMax()).append(endl);
		outBuffer.append("mean: ").append(getMean()).append(endl);
		outBuffer.append("std dev: ").append(getStandardDeviation())
				.append(endl);
		try {
			// No catch for MIAE because actual parameter is valid below
			outBuffer.append("median: ").append(getPercentile()).append(endl);
		} catch (MathIllegalStateException ex) {
			outBuffer.append("median: unavailable").append(endl);
		}
		outBuffer.append("skewness: ").append(getSkewness()).append(endl);
		outBuffer.append("kurtosis: ").append(getKurtosis()).append(endl);
		return outBuffer.toString();
	}

	/**
	 * Returns a copy of this DescriptiveStatistics instance with the same
	 * internal state.
	 * 
	 * @return a copy of this
	 */
	public DescriptiveStatisticalSummary<StorelessUnivariateStatistic> copy() {
		DescriptiveStatisticalSummary<StorelessUnivariateStatistic> result = 
				new LockfreeDescriptiveStorelessStatistics(
				 windowSize,
				 meanImpl,
				 geometricMeanImpl,
				 kurtosisImpl,
				 maxImpl,
				 minImpl,
				 percentileImpl,
				 skewnessImpl,
				 varianceImpl,
				 sumsqImpl,
				 sumImpl);
		return result;
	}
	
	/** Utility methods  */
	/**
	 * check not null for object array
	 * @param objects in an array to be checked
	 * @throws NullArgumentException in case of nulls
	 */
	private static void checkNotNull(Object... objects) throws NullArgumentException{
		MathUtils.checkNotNull(objects);
		for(Object o:objects){
			MathUtils.checkNotNull(o);
		}
	}

	/**** Disruptor specific ***/

	/**
	 * A simple data value producer that produces a double wrapped in a
	 * {@link DoubleValue} to be added to the {@link Disruptor}'s
	 * {@link RingBuffer}.
	 * <p>
	 * Note: This is not a daemon/thread and is used to just wrap the given data
	 */
	private static class DoubleValueProducerWithTranslator {

		/** A {@link Disruptor}'s {@link RingBuffer ring buffer} */
		private final RingBuffer<DoubleValue> ringBuffer;

		/**
		 * Constructor with a given ring buffer 
		 * 
		 * @param ringBuffer from disruptor
		 */
		public DoubleValueProducerWithTranslator(
				RingBuffer<DoubleValue> ringBuffer) {
			this.ringBuffer = ringBuffer;
		}

		/**
		 * A typical standard {@link EventTranslatorOneArg one argument
		 * TRANSLATOR} required to set a data to an {@link DoubleValue event}
		 */
		private static final EventTranslatorOneArg<DoubleValue, Double> TRANSLATOR = new EventTranslatorOneArg<DoubleValue, Double>() {
			public void translateTo(DoubleValue event, long sequence,
					Double data) {
				event.setValue(data);
			}
		};

		/**
		 * Publishes data to a ring buffer.
		 * 
		 * @param data a value
		 */
		public void onData(Double data) {
			ringBuffer.publishEvent(TRANSLATOR, data);
		}
	}

	/**
	 * An Event data class to operate within {@link Disruptor}
	 */
	private static class DoubleValue {
		/** value */
		double value;
		/**
		 * value is returned
		 * @return value
		 */
		public double getValue() {
			return value;
		}
		/** 
		 * set the value
		 * @param val to be set
		 * @return this instance
		 */
		public DoubleValue setValue(double val) {
			this.value = val;
			return this;
		}
		/** 
		 * a class for factory to generating events
		 */
		public static class FACTORY implements EventFactory<DoubleValue> {
			/**
			 * a new instance method
			 */
			public DoubleValue newInstance() {
				return new DoubleValue();
			}
		}
	}

	/**
	 * A general consumer of events that incrementally adds a data received from
	 * {@link Disruptor}'s {@link RingBuffer ring buffer} to a
	 * {@link StorelessUnivariateStatistic}
	 * 
	 * @param <S> an implement of {@link StorelessUnivariateStatistic}
	 */
	private static class StatisticEventHandler<S extends StorelessUnivariateStatistic>
			implements EventHandler<DoubleValue> {

		/**
		 * A {@link StorelessUnivariateStatistic} instance to be computed for
		 * data set published on a {@link RingBuffer ring buffer}
		 */
		private final S stats;

		/**
		 * Constructor
		 * 
		 * @param stats to be used for incrementally building
		 */
		public StatisticEventHandler(S stats) {
			this.stats = stats;
		}
		
		/**
		 *  a creation method for array of handlers.
		 *  @param storelessStats an array of stats
		 *  @return StatisticEventHandler[]
		 *  <T> a type of {@code StorelessUnivariateStatistic}
		 */
		public  static <T extends StorelessUnivariateStatistic> StatisticEventHandler<T>[] create(
				T[] storelessStats) {
			@SuppressWarnings("unchecked")
			StatisticEventHandler<T>[] storelessStatEventHandlers= new StatisticEventHandler[storelessStats.length];
			for (int i = 0; i < storelessStats.length; i++){
				storelessStatEventHandlers[i] = new StatisticEventHandler<T>(storelessStats[i]);
			}
			return storelessStatEventHandlers;
		}

		/**
		 * {@inheritDoc}. This event contains a double value to be added to
		 * stats
		 */
		public  void onEvent(DoubleValue event, long sequence, boolean endOfBatch)
				throws Exception {
			final double value=event.getValue();
			stats.increment(value);
		}
	}
}
