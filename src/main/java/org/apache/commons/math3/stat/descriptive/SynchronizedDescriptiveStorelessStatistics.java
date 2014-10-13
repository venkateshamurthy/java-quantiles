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

import org.apache.commons.math3.exception.MathIllegalArgumentException;
import org.apache.commons.math3.exception.MathIllegalStateException;

/**
 * Implementation of
 * {@link org.apache.commons.math3.stat.descriptive.DescriptiveStatisticalSummary}
 * that is safe to use in a multithreaded environment. Multiple threads can
 * safely operate on a single instance without causing runtime exceptions due to
 * race conditions. In effect, this implementation makes modification and access
 * methods atomic operations for a single instance. That is to say, as one
 * thread is computing a statistic from the instance, no other thread can modify
 * the instance nor compute another statistic.
 * @since 3.4
 */
public class SynchronizedDescriptiveStorelessStatistics implements
        DescriptiveStatisticalSummary<StorelessUnivariateStatistic> {
	/** Core instance to be decorated */
	private final DescriptiveStorelessStatistics stats;

	/**
	 * Default constructor
	 */
	public SynchronizedDescriptiveStorelessStatistics() {
		this(new DescriptiveStorelessStatistics());
	}

	/**
	 * Constructor
	 * @param stats to be lock synchronized
	 */
	public SynchronizedDescriptiveStorelessStatistics(
	        DescriptiveStorelessStatistics stats) {
		this.stats = stats;
	}

	/**
	 * {@inheritDoc}
	 */
	public synchronized void addValue(double v) {
		stats.addValue(v);
	}

	/**
	 * Returns the <a href="http://www.xycoon.com/arithmetic_mean.htm">
	 * arithmetic mean </a> of the available values
	 * @return The mean or Double.NaN if no values have been added.
	 */
	public synchronized double getMean() {
		return stats.getMean();
	}

	/**
	 * Returns the <a href="http://www.xycoon.com/geometric_mean.htm"> geometric
	 * mean </a> of the available values
	 * @return The geometricMean, Double.NaN if no values have been added, or if
	 *         the product of the available values is less than or equal to 0.
	 */
	public synchronized double getGeometricMean() {
		return stats.getGeometricMean();
	}

	/**
	 * Returns the (sample) variance of the available values.
	 * <p>
	 * This method returns the bias-corrected sample variance (using
	 * {@code n - 1} in the denominator). Use {@link #getPopulationVariance()}
	 * for the non-bias-corrected population variance.
	 * </p>
	 * @return The variance, Double.NaN if no values have been added or 0.0 for a
	 *         single value set.
	 */
	public synchronized double getVariance() {
		return stats.getVariance();
	}

	/**
	 * Returns the standard deviation of the available values.
	 * @return The standard deviation, Double.NaN if no values have been added or
	 *         0.0 for a single value set.
	 */
	public synchronized double getStandardDeviation() {
		return stats.getStandardDeviation();
	}

	/**
	 * Returns the skewness of the available values. Skewness is a measure of
	 * the asymmetry of a given distribution.
	 * @return The skewness, Double.NaN if no values have been added or 0.0 for a
	 *         value set &lt;=2.
	 */
	public synchronized double getSkewness() {
		return stats.getSkewness();
	}

	/**
	 * Returns the Kurtosis of the available values. Kurtosis is a measure of
	 * the "peakedness" of a distribution
	 * @return The kurtosis, Double.NaN if no values have been added, or 0.0 for
	 *         a value set &lt;=3.
	 */
	public synchronized double getKurtosis() {
		return stats.getKurtosis();
	}

	/**
	 * Returns the maximum of the available values
	 * @return The max or Double.NaN if no values have been added.
	 */
	public synchronized double getMax() {
		return stats.getMax();
	}

	/**
	 * Returns the minimum of the available values
	 * @return The min or Double.NaN if no values have been added.
	 */
	public synchronized double getMin() {
		return stats.getMin();
	}

	/**
	 * Returns the number of values passed.
	 * @return The number of available values
	 */
	public synchronized long getN() {
		return stats.getN();
	}

	/**
	 * Returns the sum of the values that have been added to Univariate.
	 * @return The sum or Double.NaN if no values have been added
	 */
	public synchronized double getSum() {
		return stats.getSum();
	}

	/**
	 * Returns the sum of the squares of the available values.
	 * @return The sum of the squares or Double.NaN if no values have been added.
	 */
	public synchronized double getSumsq() {
		return stats.getSumsq();
	}

	/**
	 * Resets all statistics and storage
	 */
	public synchronized void clear() {
		stats.clear();
	}

	/** {@inheritDoc} */
	public synchronized double getPercentile()
	        throws MathIllegalStateException, MathIllegalArgumentException {
		return stats.getPercentile();
	}

	/** {@inheritDoc} */
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

	/** {@inheritDoc} */
	public synchronized DescriptiveStatisticalSummary<StorelessUnivariateStatistic> withPercentile(
	        StorelessUnivariateStatistic percentileImpl) {
		return stats.withPercentile(percentileImpl);
	}

	/** {@inheritDoc} */
	public synchronized DescriptiveStatisticalSummary<StorelessUnivariateStatistic> withMean(
	        StorelessUnivariateStatistic meanImpl) {
		return stats.withMean(meanImpl);
	}

	/** {@inheritDoc} */
	public synchronized DescriptiveStatisticalSummary<StorelessUnivariateStatistic> withKurtosis(
	        StorelessUnivariateStatistic kurtosisImpl) {
		return stats.withKurtosis(kurtosisImpl);
	}

	/** {@inheritDoc} */
	public synchronized DescriptiveStatisticalSummary<StorelessUnivariateStatistic> withVariance(
	        StorelessUnivariateStatistic varianceImpl) {
		return stats.withVariance(varianceImpl);
	}

	/** {@inheritDoc} */
	public synchronized DescriptiveStatisticalSummary<StorelessUnivariateStatistic> withSkewness(
	        StorelessUnivariateStatistic skewnessImpl) {
		return stats.withSkewness(skewnessImpl);
	}

	/** {@inheritDoc} */
	public synchronized DescriptiveStatisticalSummary<StorelessUnivariateStatistic> copy() {
		return stats.copy();
	}

	/** {@inheritDoc} */
	public void halt() {
		// Do nothing as there are no active elements such as threads in this
		// instance
	}
}
