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
import java.util.concurrent.atomic.AtomicLong;

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
import org.apache.commons.math3.util.MathUtils;

/**
 * A mirror implementation of methods in {@link DescriptiveStatistics} but for store-less statistics
 */
public class DescriptiveStorelessStatistics implements
		DescriptiveStatisticalSummary<StorelessUnivariateStatistic>, Serializable {

	/** Serialization UID */
	private static final long serialVersionUID = 4133067267405273064L;

	/** {@code StorelessUnivariateStatistic Statistic implementations} */
	private final StorelessUnivariateStatistic countImpl;
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
	 * Construct a DescriptiveStatistics instance
	 */
	public DescriptiveStorelessStatistics() {
		this(new Count(), new Mean(), new GeometricMean(), new Kurtosis(),
				new Max(), new Min(), new PSquarePercentile(50d),
				new Skewness(), new Variance(), new SumOfSquares(), new Sum());
	}

	/**
	 * Construct a DescriptiveStatistics instance with an infinite window and
	 * the initial data values in double[] initialDoubleArray. If
	 * initialDoubleArray is null, then this constructor corresponds to
	 * DescriptiveStatistics()
	 * 
	 * @param initialDoubleArray  the initial double[].
	 */
	public DescriptiveStorelessStatistics(double[] initialDoubleArray) {
		this();
		if (initialDoubleArray != null) {
			for (double d : initialDoubleArray){
				addValue(d);
			}
		}
	}
	/**
	 * Constructor
	 * @param count counter
	 * @param mean to compute average
	 * @param geometricMean to compute geometric average
	 * @param kurtosis to compute kurtosis
	 * @param max
	 * @param min
	 * @param percentile
	 * @param skewness
	 * @param variance
	 * @param sumsq
	 * @param sum
	 * @throws MathIllegalArgumentException
	 */
	private DescriptiveStorelessStatistics(StorelessUnivariateStatistic count,
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
				this.countImpl = count, this.meanImpl = mean,
				this.geometricMeanImpl = geometricMean,
				this.kurtosisImpl = kurtosis, this.maxImpl = max,
				this.minImpl = min,this.percentileImpl = percentile,
				this.skewnessImpl = skewness,
				this.varianceImpl = variance, this.sumsqImpl = sumsq,
				this.sumImpl = sum };
		checkNotNull((Object[])storelessStats);
	}
	/**
	 * Returns a copy of this DescriptiveStatistics instance with the same
	 * internal state.
	 * 
	 * @return a copy of this
	 */
	public DescriptiveStorelessStatistics copy() {
		DescriptiveStorelessStatistics result = new DescriptiveStorelessStatistics(
				 countImpl,
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


	/**
	 * Adds the value to the dataset. If the dataset is at the maximum size
	 * (i.e., the number of stored elements equals the currently configured
	 * windowSize), the first (oldest) element in the dataset is discarded to
	 * make room for the new value.
	 * 
	 * @param value to be added
	 */
	public void addValue(final double value) {
		for (StorelessUnivariateStatistic stat : storelessStats){
			stat.increment(value);
		}
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
	 * <p>
	 * This method returns the bias-corrected sample variance (using
	 * {@code n - 1} in the denominator). Use {@link #getPopulationVariance()}
	 * for the non-bias-corrected population variance.
	 * </p>
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
	 * Returns the number of values passed.
	 * 
	 * @return The number of available values
	 */
	public long getN() {
		return countImpl.getN();
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
		for (StorelessUnivariateStatistic s : storelessStats){
			s.clear();
		}
	}

	/** 
	 * Get the Percentile
	 * @return percentile computed thus far
	 */
	public double getPercentile()  {
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
	 * A Count class
	 * 
	 */
	private static class Count extends AbstractStorelessUnivariateStatistic {
		/** a counter */
		private final AtomicLong countLong;
		/**
		 * Constructor
		 */
		public Count() {
			this(0);
		}
		/** Constructor with a count*/
		public Count(long countLong) {
			super();
			this.countLong = new AtomicLong(countLong);
		}
		/** 
		 * gets the count counted 
		 * @return count value
		 */
		public long getN() {
			return countLong.get();
		}
		/** {@inheritDoc}*/
		@Override
		public void increment(double d) {
			countLong.incrementAndGet();
		}
		/** {@inheritDoc}*/
		@Override
		public double getResult() {
			return countLong.get();
		}
		/** {@inheritDoc}*/
		@Override
		public StorelessUnivariateStatistic copy() {
			return new Count(countLong.get());
		}
		/** {@inheritDoc}*/
		@Override
		public void clear() {
			countLong.set(0L);
		}
	}
	/** {@inheritDoc}*/
	public DescriptiveStatisticalSummary<StorelessUnivariateStatistic> withPercentile(
			StorelessUnivariateStatistic percentile) {
		return new DescriptiveStorelessStatistics(countImpl, meanImpl,
				geometricMeanImpl, kurtosisImpl, maxImpl, minImpl,
				percentile, skewnessImpl, varianceImpl, sumsqImpl, sumImpl);
	}
	/** {@inheritDoc}*/
	public DescriptiveStatisticalSummary<StorelessUnivariateStatistic> withMean(
			StorelessUnivariateStatistic mean) {
		return new DescriptiveStorelessStatistics(countImpl, mean,
				geometricMeanImpl, kurtosisImpl, maxImpl, minImpl,
				percentileImpl, skewnessImpl, varianceImpl, sumsqImpl, sumImpl);
	}
	/** {@inheritDoc}*/
	public DescriptiveStatisticalSummary<StorelessUnivariateStatistic> withKurtosis(
			StorelessUnivariateStatistic kurtosis) {
		return new DescriptiveStorelessStatistics(countImpl, meanImpl,
				geometricMeanImpl, kurtosis, maxImpl, minImpl,
				percentileImpl, skewnessImpl, varianceImpl, sumsqImpl, sumImpl);
	}
	/** {@inheritDoc}*/
	public DescriptiveStatisticalSummary<StorelessUnivariateStatistic> withVariance(
			StorelessUnivariateStatistic variance) {
		return new DescriptiveStorelessStatistics(countImpl, meanImpl,
				geometricMeanImpl, kurtosisImpl, maxImpl, minImpl,
				percentileImpl, skewnessImpl, variance, sumsqImpl, sumImpl);
	}
	/** {@inheritDoc}*/
	public DescriptiveStatisticalSummary<StorelessUnivariateStatistic> withSkewness(
			StorelessUnivariateStatistic skewness) {
		return new DescriptiveStorelessStatistics(countImpl, meanImpl,
				geometricMeanImpl, kurtosisImpl, maxImpl, minImpl,
				percentileImpl, skewness, varianceImpl, sumsqImpl, sumImpl);
	}
	
	/**{@inheritDoc}*/
	public void halt() {
	}

	/** Utility methods  */
	/**
	 * Utility to check null objects in an array
	 * @param objects
	 * @throws NullArgumentException
	 */
	private static void checkNotNull(Object... objects) throws NullArgumentException{
		MathUtils.checkNotNull(objects);
		for(Object o:objects){
			MathUtils.checkNotNull(o);
		}
	}

}
