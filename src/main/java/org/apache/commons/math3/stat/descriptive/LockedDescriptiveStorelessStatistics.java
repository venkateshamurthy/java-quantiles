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


import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock.ReadLock;
import java.util.concurrent.locks.ReentrantReadWriteLock.WriteLock;

import org.apache.commons.math3.exception.MathIllegalArgumentException;
import org.apache.commons.math3.exception.MathIllegalStateException;

/**
 * Implementation of {@link org.apache.commons.math3.stat.descriptive.DescriptiveStatisticalSummary}
 * that is safe to use in a multithreaded environment. Multiple threads can
 * safely operate on a single instance without causing runtime exceptions due to
 * race conditions. In effect, this implementation makes modifications as atomic 
 * but access methods as concurrent operations for a single instance. 
 * That is to say, as one thread is computing a statistic from the instance,
 * no other thread can modify the instance however can compute another statistic.
 * 
 * @since 3.4
 */
public class LockedDescriptiveStorelessStatistics implements
		DescriptiveStatisticalSummary<StorelessUnivariateStatistic> {
	/** Lock */
	private final ReadWriteLock lock = new ReentrantReadWriteLock();;
	/** read lock */
	private final ReadLock rLock = (ReadLock) lock.readLock();
	/*** write lock */
	private final WriteLock wLock = (WriteLock) lock.writeLock();
	/** Core instance to be decorated */
	private final DescriptiveStorelessStatistics stats;
	/**
	 * Default constructor
	 */
	public LockedDescriptiveStorelessStatistics() {
		this(new DescriptiveStorelessStatistics());
	}
	/**
	 * Constructor
	 * @param stats to be lock synchronized
	 */
	public LockedDescriptiveStorelessStatistics(
			DescriptiveStorelessStatistics stats) {
		this.stats = stats;
	}

	/**
	 * {@inheritDoc}
	 */
	public void addValue(double v) {
		wLock.lock();
		try {
			stats.addValue(v);
		} finally {
			wLock.unlock();
		}
	}

	/**
	 * Returns the <a href="http://www.xycoon.com/arithmetic_mean.htm">
	 * arithmetic mean </a> of the available values
	 * 
	 * @return The mean or Double.NaN if no values have been added.
	 */
	public double getMean() {
		rLock.lock();
		try {
			return stats.getMean();
		} finally {
			rLock.unlock();
		}
	}

	/**
	 * Returns the <a href="http://www.xycoon.com/geometric_mean.htm"> geometric
	 * mean </a> of the available values
	 * 
	 * @return The geometricMean, Double.NaN if no values have been added, or if
	 *         the product of the available values is less than or equal to 0.
	 */
	public double getGeometricMean() {
		rLock.lock();
		try {
			return stats.getGeometricMean();
		} finally {
			rLock.unlock();
		}
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
		rLock.lock();
		try {
			return stats.getVariance();
		} finally {
			rLock.unlock();
		}
	}

	/**
	 * Returns the standard deviation of the available values.
	 * 
	 * @return The standard deviation, Double.NaN if no values have been added
	 *         or 0.0 for a single value set.
	 */
	public double getStandardDeviation() {
		rLock.lock();
		try {
			return stats.getStandardDeviation();
		} finally {
			rLock.unlock();
		}
	}

	/**
	 * Returns the skewness of the available values. Skewness is a measure of
	 * the asymmetry of a given distribution.
	 * 
	 * @return The skewness, Double.NaN if no values have been added or 0.0 for
	 *         a value set &lt;=2.
	 */
	public double getSkewness() {
		rLock.lock();
		try {
			return stats.getSkewness();
		} finally {
			rLock.unlock();
		}
	}

	/**
	 * Returns the Kurtosis of the available values. Kurtosis is a measure of
	 * the "peakedness" of a distribution
	 * 
	 * @return The kurtosis, Double.NaN if no values have been added, or 0.0 for
	 *         a value set &lt;=3.
	 */
	public double getKurtosis() {
		rLock.lock();
		try {
			return stats.getKurtosis();
		} finally {
			rLock.unlock();
		}
	}

	/**
	 * Returns the maximum of the available values
	 * 
	 * @return The max or Double.NaN if no values have been added.
	 */
	public double getMax() {
		rLock.lock();
		try {
			return stats.getMax();
		} finally {
			rLock.unlock();
		}
	}

	/**
	 * Returns the minimum of the available values
	 * 
	 * @return The min or Double.NaN if no values have been added.
	 */
	public double getMin() {
		rLock.lock();
		try {
			return stats.getMin();
		} finally {
			rLock.unlock();
		}
	}

	/**
	 * Returns the number of values passed.
	 * 
	 * @return The number of available values
	 */
	public long getN() {
		rLock.lock();
		try {
			return stats.getN();
		} finally {
			rLock.unlock();
		}
	}

	/**
	 * Returns the sum of the values that have been added to Univariate.
	 * 
	 * @return The sum or Double.NaN if no values have been added
	 */
	public double getSum() {
		rLock.lock();
		try {
			return stats.getSum();
		} finally {
			rLock.unlock();
		}
	}

	/**
	 * Returns the sum of the squares of the available values.
	 * 
	 * @return The sum of the squares or Double.NaN if no values have been
	 *         added.
	 */
	public double getSumsq() {
		rLock.lock();
		try {
			return stats.getSumsq();
		} finally {
			rLock.unlock();
		}
	}

	/**
	 * Resets all statistics and storage
	 */
	public void clear() {
		wLock.lock();
		try {
			stats.clear();
		} finally {
			wLock.unlock();
		}
	}
	/** {@inheritDoc}*/
	public double getPercentile() throws MathIllegalStateException,
			MathIllegalArgumentException {
		rLock.lock();
		try {
			return stats.getPercentile();
		} finally {
			rLock.unlock();
		}
	}

	/** {@inheritDoc}*/
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

	/** {@inheritDoc}*/
	public DescriptiveStatisticalSummary<StorelessUnivariateStatistic> withPercentile(
			StorelessUnivariateStatistic percentileImpl) {
		wLock.lock();
		try {
			return stats.withPercentile(percentileImpl);
		} finally {	wLock.unlock();	}
	}
	
	/** {@inheritDoc}*/
	public DescriptiveStatisticalSummary<StorelessUnivariateStatistic> withMean(
			StorelessUnivariateStatistic meanImpl) {
		wLock.lock();
		try {
			return stats.withMean(meanImpl);
		} finally {	wLock.unlock();	}
	}
	
	/** {@inheritDoc}*/
	public DescriptiveStatisticalSummary<StorelessUnivariateStatistic> withKurtosis(
			StorelessUnivariateStatistic kurtosisImpl) {
		wLock.lock();
		try {
			return stats.withKurtosis(kurtosisImpl);
		} finally {	wLock.unlock();	}
	}
	
	/** {@inheritDoc}*/
	public DescriptiveStatisticalSummary<StorelessUnivariateStatistic> withVariance(
			StorelessUnivariateStatistic varianceImpl) {
		wLock.lock();
		try {
			return stats.withVariance(varianceImpl);
		} finally {	wLock.unlock();	}
	}
	
	/** {@inheritDoc}*/
	public DescriptiveStatisticalSummary<StorelessUnivariateStatistic> withSkewness(
			StorelessUnivariateStatistic skewnessImpl) {
		wLock.lock();
		try {
			return stats.withSkewness(skewnessImpl);
		} finally {	wLock.unlock();	}
	}
	
	/** {@inheritDoc}*/
	public DescriptiveStatisticalSummary<StorelessUnivariateStatistic> copy() {
		wLock.lock();
		try {
			return stats.copy();
		} finally {	wLock.unlock();	}
	}
	
	/** {@inheritDoc}*/
	public void halt() {
		//Do nothing as there are no active elements such as threads in this instance
	}

}
