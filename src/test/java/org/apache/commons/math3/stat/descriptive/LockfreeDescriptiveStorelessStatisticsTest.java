/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See the NOTICE file distributed with this
 * work for additional information regarding copyright ownership. The ASF
 * licenses this file to You under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law
 * or agreed to in writing, software distributed under the License is
 * distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the specific language
 * governing permissions and limitations under the License.
 */
package org.apache.commons.math3.stat.descriptive;

import java.util.Locale;
import java.util.concurrent.locks.LockSupport;

import org.apache.commons.math3.TestUtils;
import org.apache.commons.math3.exception.MathIllegalArgumentException;
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
import org.junit.Assert;
import org.junit.Test;

/**
 * Test cases for the {@link LockfreeDescriptiveStorelessStatisticsTest} class.
 *
 */
public class LockfreeDescriptiveStorelessStatisticsTest {

    protected DescriptiveStatisticalSummary<StorelessUnivariateStatistic> createDescriptiveStatistics() {
        return new LockfreeDescriptiveStorelessStatistics();
    }
    
    protected DescriptiveStatisticalSummary<StorelessUnivariateStatistic> createDescriptiveStatistics(
    		DescriptiveStatisticalSummary<StorelessUnivariateStatistic> copy) {
        return copy.copy();
    }

    @Test
    public void testSetterInjection() {
    	DescriptiveStatisticalSummary<StorelessUnivariateStatistic> stats = createDescriptiveStatistics();
        stats.addValue(1);
        stats.addValue(3);
        LockSupport.parkNanos(10000L);
        Assert.assertEquals(2, stats.getMean(), 1E-10);
        // Now lets try some new math
        stats=stats.withMean(new deepMean());
        Assert.assertEquals(42, stats.getMean(), 1E-10);
    }

    @Test
    public void testCopy() {
    	DescriptiveStatisticalSummary<StorelessUnivariateStatistic> stats = createDescriptiveStatistics();
        stats.addValue(1);
        stats.addValue(3);
        DescriptiveStatisticalSummary<StorelessUnivariateStatistic> copy =createDescriptiveStatistics(stats);
        Assert.assertEquals(2, copy.getMean(), 1E-10);
        // Now lets try some new math
        stats= stats.withMean(new deepMean());
        copy = stats.copy();
        Assert.assertEquals(42, copy.getMean(), 1E-10);
    }

    @Test
    public void testToString() {
        DescriptiveStatisticalSummary<StorelessUnivariateStatistic> stats = createDescriptiveStatistics();
        stats.addValue(1);
        stats.addValue(2);
        stats.addValue(3);
        LockSupport.parkNanos(10000L);
        Locale d = Locale.getDefault();
        Locale.setDefault(Locale.US);
        Assert.assertEquals("LockfreeDescriptiveStorelessStatistics:\n" +
                     "n: 3\n" +
                     "min: 1.0\n" +
                     "max: 3.0\n" +
                     "mean: 2.0\n" +
                     "std dev: 1.0\n" +
                     "median: 2.0\n" +
                     "skewness: 0.0\n" +
                     "kurtosis: NaN\n",  stats.toString());
        Locale.setDefault(d);
    }
    

    @Test
    public void test20090720() {
        DescriptiveStatisticalSummary<StorelessUnivariateStatistic> descriptiveStatistics = createDescriptiveStatistics();
        for (int i = 0; i < 161; i++) {
            descriptiveStatistics.addValue(1.2);
        }
        descriptiveStatistics.clear();
        descriptiveStatistics.addValue(1.2);
        Assert.assertEquals(1, descriptiveStatistics.getN());
    }

    
    @Test
    public void testSummaryConsistency() {
        final DescriptiveStatistics dstats = new DescriptiveStatistics();
        
        final double tol = 1E-12;
        for (int i = 0; i < 20; i++) {
            dstats.addValue(i);
            double[] values = dstats.getValues();
            LockfreeDescriptiveStorelessStatistics descriptiveStorelessStats = new LockfreeDescriptiveStorelessStatistics();
            for(double d:values){
            	descriptiveStorelessStats.addValue(d);
            	LockSupport.parkNanos(1000L);
            }
            String msg="Failing at i="+i;
            TestUtils.assertEquals(msg,dstats.getMean(), descriptiveStorelessStats.getMean(), tol);
            TestUtils.assertEquals(msg,new Mean().evaluate(values), dstats.getMean(), tol);
            TestUtils.assertEquals(msg,dstats.getMax(), descriptiveStorelessStats.getMax(), tol);
            TestUtils.assertEquals(msg,new Max().evaluate(values), dstats.getMax(), tol);
            TestUtils.assertEquals(msg,dstats.getGeometricMean(), descriptiveStorelessStats.getGeometricMean(), tol);
            TestUtils.assertEquals(msg,new GeometricMean().evaluate(values), dstats.getGeometricMean(), tol);
            TestUtils.assertEquals(msg,dstats.getMin(), descriptiveStorelessStats.getMin(), tol);
            TestUtils.assertEquals(msg,new Min().evaluate(values), dstats.getMin(), tol);
            TestUtils.assertEquals(msg,dstats.getStandardDeviation(), descriptiveStorelessStats.getStandardDeviation(), tol);
            TestUtils.assertEquals(msg,dstats.getVariance(), descriptiveStorelessStats.getVariance(), tol);
            TestUtils.assertEquals(msg,new Variance().evaluate(values), dstats.getVariance(), tol);
            TestUtils.assertEquals(msg,dstats.getSum(), descriptiveStorelessStats.getSum(), tol);
            TestUtils.assertEquals(msg,new Sum().evaluate(values), dstats.getSum(), tol);
            TestUtils.assertEquals(msg,dstats.getSumsq(), descriptiveStorelessStats.getSumsq(), tol);
            TestUtils.assertEquals(msg,new SumOfSquares().evaluate(values), dstats.getSumsq(), tol);
           // TestUtils.assertEquals(dstats.getPopulationVariance(), descriptiveStorelessStats.getPopulationVariance(), tol);
            TestUtils.assertEquals(msg,new Variance(false).evaluate(values), dstats.getPopulationVariance(), tol);
            descriptiveStorelessStats.clear();
            descriptiveStorelessStats.halt();
            LockSupport.parkNanos(10000L);
        }
    }

    @Test
	public void testWithFunctions() {
		final double tol = 1E-12;
		final double[] data = new double[] {
				-0.012086732064244697,
				-0.24975668704012527,
				0.5706168483164684,
				-0.322111769955327,
				0.24166759508327315,
				// Double.NaN,
				// Double.POSITIVE_INFINITY,
				0.16698443218942854, -0.10427763937565114,
				-0.15595963093172435, -0.028075857595882995,
				-0.24137994506058857, 0.47543170476574426,
				-0.07495595384947631, 0.37445697625436497, -0.09944199541668033 };

		final DescriptiveStatisticalSummary<StorelessUnivariateStatistic> ds = createDescriptiveStatistics();
		final DescriptiveStatisticalSummary<StorelessUnivariateStatistic> dsMean = createDescriptiveStatistics()
				.withMean(new Mean());
		final DescriptiveStatisticalSummary<StorelessUnivariateStatistic> dsKurt = createDescriptiveStatistics()
				.withKurtosis(new Kurtosis());
		final DescriptiveStatisticalSummary<StorelessUnivariateStatistic> dsSkew = createDescriptiveStatistics()
				.withSkewness(new Skewness());
		final DescriptiveStatisticalSummary<StorelessUnivariateStatistic> ds50 = createDescriptiveStatistics()
				.withPercentile(new PSquarePercentile(50d));
		final DescriptiveStatisticalSummary<StorelessUnivariateStatistic> dsVar = createDescriptiveStatistics()
				.withVariance(new Variance());
		for (double d : data) {
			ds.addValue(d);
			dsMean.addValue(d);
			ds50.addValue(d);
			dsKurt.addValue(d);
			dsSkew.addValue(d);
			dsVar.addValue(d);
			LockSupport.parkNanos(10000L); // this is needed as otherwise Lockfree version doesnt work
		}
		LockSupport.parkNanos(10000L); // this is needed as otherwise Lockfree version doesnt work
		TestUtils.assertEquals(ds.getMean(), dsMean.getMean(), tol);
		TestUtils.assertEquals(ds.getMean(), dsKurt.getMean(), tol);
		TestUtils.assertEquals(ds.getMean(), dsSkew.getMean(), tol);
		TestUtils.assertEquals(ds.getMean(), ds50.getMean(), tol);
		TestUtils.assertEquals(ds.getMean(), dsVar.getMean(), tol);
		TestUtils.assertEquals(ds.getMean(), dsKurt.getMean(), tol);
		TestUtils.assertEquals(ds.getKurtosis(), dsKurt.getKurtosis(), tol);
		TestUtils.assertEquals(ds.getSkewness(), dsSkew.getSkewness(), tol);
		TestUtils.assertEquals(ds.getPercentile(), ds50.getPercentile(), tol);
		TestUtils.assertEquals(ds.getVariance(), dsVar.getVariance(), tol);
	}

    @Test
    public void testMath1129(){
        final double[] data = new double[] {
            -0.012086732064244697,
            -0.24975668704012527,
            0.5706168483164684,
            -0.322111769955327,
            0.24166759508327315,
            //Double.NaN,
            0.16698443218942854,
            -0.10427763937565114,
            -0.15595963093172435,
            -0.028075857595882995,
            -0.24137994506058857,
            0.47543170476574426,
            -0.07495595384947631,
            0.37445697625436497,
            -0.09944199541668033
        };
        
        final DescriptiveStatisticalSummary<StorelessUnivariateStatistic> ds = 
        		new LockfreeDescriptiveStorelessStatistics();
        final DescriptiveStatisticalSummary<StorelessUnivariateStatistic> ds50 = 
        		new LockfreeDescriptiveStorelessStatistics((data));
        final DescriptiveStatisticalSummary<StorelessUnivariateStatistic> ds75 = 
        		ds.withPercentile(new PSquarePercentile(75d));
        final DescriptiveStatisticalSummary<StorelessUnivariateStatistic> ds25 = 
        		ds.withPercentile(new PSquarePercentile(25d));
        for(double d:data){
        	ds.addValue(d);
        	ds75.addValue(d);
        	ds25.addValue(d);
        	LockSupport.parkNanos(10000);
        }
        
        Assert.assertEquals(ds50.getPercentile(), ds.getPercentile(),1e-05);
        final double t = ds75.getPercentile();
        final double o = ds25.getPercentile();

        final double iqr = t - o;
        // System.out.println(String.format("25th percentile %s 75th percentile %s", o, t));
        Assert.assertTrue(iqr >= 0);
    }

    

    // Test UnivariateStatistics impls for setter injection tests

    /**
     * A new way to compute the mean
     */
    static class deepMean implements StorelessUnivariateStatistic {

        public double evaluate(double[] values, int begin, int length) {
            return 42;
        }

        public double evaluate(double[] values) {
            return 42;
        }
        public StorelessUnivariateStatistic copy() {
            return new deepMean();
        }

		public void increment(double d) {
						
		}

		public void incrementAll(double[] values)
				throws MathIllegalArgumentException {
			
		}

		public void incrementAll(double[] values, int start, int length)
				throws MathIllegalArgumentException {
			
		}

		public double getResult() {
			return 42;
		}

		public long getN() {
			return 0;
		}

		public void clear() {
			
		}
    }
}
