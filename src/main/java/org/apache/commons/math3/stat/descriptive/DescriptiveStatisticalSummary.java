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


/**
 * An extended interface to {@link StatisticalSummary} that has other important additional public
 * functions of {@link DescriptiveStatistics} . This helps in viewing Store and Storeless variant uniformly.
 * 
 * @param <S> is one of {@link UnivariateStatistic},{@link StorelessUnivariateStatistic}
 */
public interface DescriptiveStatisticalSummary<S extends UnivariateStatistic>
		extends StatisticalSummary {
	
	/**
	 * adds a value
	 * 
	 * @param value passed in
	 */
	void addValue(double value );

	/** 
	 * clear 
	 */
	void clear();

	/**
	 * Return Percentile
	 * 
	 * @return Percentile value
	 */
	double getPercentile();

	/**
	 * Return Sum of squares
	 * 
	 * @return Sum of squares value
	 */
	double getSumsq();

	/**
	 * Return Geometric mean
	 * 
	 * @return Geometric value
	 */
	double getGeometricMean();

	/**
	 * Return Skewness
	 * 
	 * @return Skewness value
	 */
	double getSkewness();

	/**
	 * Return Kurtosis
	 * 
	 * @return kurtosis value
	 */
	double getKurtosis();

	/**
	 * Return an instance with a given {@link org.apache.commons.math3.stat.descriptive.rank.Percentile Store} or
	 * {@link org.apache.commons.math3.stat.descriptive.rank.PSquarePercentile Storeless} version of Percentile
	 * 
	 * @param percentile an instance of {@code Percentile}/{@code PSquarePercentile} to be set
	 * @return DescriptiveStatisticalSummary
	 */
	DescriptiveStatisticalSummary<S> withPercentile(S percentile);

	/**
	 * Return an instance with a given {@link org.apache.commons.math3.stat.descriptive.moment.Mean}
	 * 
	 * @param mean to be set
	 * @return DescriptiveStatisticalSummary
	 */
	DescriptiveStatisticalSummary<S> withMean(S mean);

	/**
	 * Return an instance with a given {@link org.apache.commons.math3.stat.descriptive.moment.Kurtosis}
	 * 
	 * @param kurtosis to be set
	 * @return DescriptiveStatisticalSummary
	 */
	DescriptiveStatisticalSummary<S> withKurtosis(S kurtosis);

	/**
	 * Return an instance with a given {@link org.apache.commons.math3.stat.descriptive.moment.Variance}
	 * 
	 * @param variance  to be set
	 * @return DescriptiveStatisticalSummary
	 */
	DescriptiveStatisticalSummary<S> withVariance(S variance);

	/**
	 * Return an instance with a given {@link org.apache.commons.math3.stat.descriptive.moment.Skewness}
	 * 
	 * @param skewness to be set
	 * @return DescriptiveStatisticalSummary
	 */
	DescriptiveStatisticalSummary<S> withSkewness(S skewness);
	
	/**
	 * Halt method to stop any process/thread handles within the instance
	 */
	void halt();
	
	/**
	 * A copy method
	 * @return copy of this instance
	 */
	DescriptiveStatisticalSummary<S> copy();
}
