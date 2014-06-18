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
package org.apache.commons.math3.stat.descriptive.rank;
import static org.apache.commons.math3.stat.descriptive.rank.Percentile.Type.CM;
import static org.apache.commons.math3.stat.descriptive.rank.Percentile.Type.R_1;
import static org.apache.commons.math3.stat.descriptive.rank.Percentile.Type.R_2;
import static org.apache.commons.math3.stat.descriptive.rank.Percentile.Type.R_3;
import static org.apache.commons.math3.stat.descriptive.rank.Percentile.Type.R_4;
import static org.apache.commons.math3.stat.descriptive.rank.Percentile.Type.R_5;
import static org.apache.commons.math3.stat.descriptive.rank.Percentile.Type.R_6;
import static org.apache.commons.math3.stat.descriptive.rank.Percentile.Type.R_7;
import static org.apache.commons.math3.stat.descriptive.rank.Percentile.Type.R_8;
import static org.apache.commons.math3.stat.descriptive.rank.Percentile.Type.R_9;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.exception.MathIllegalArgumentException;
import org.apache.commons.math3.exception.MathUnsupportedOperationException;
import org.apache.commons.math3.exception.NotANumberException;
import org.apache.commons.math3.exception.NullArgumentException;
import org.apache.commons.math3.exception.OutOfRangeException;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.stat.descriptive.UnivariateStatistic;
import org.apache.commons.math3.stat.descriptive.UnivariateStatisticAbstractTest;
import org.apache.commons.math3.stat.descriptive.rank.Percentile.PivotingStrategy;
import org.apache.commons.math3.stat.descriptive.rank.Percentile.Type;
import org.apache.commons.math3.stat.ranking.NaNStrategy;
import org.apache.commons.math3.util.MathArrays;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

/**
 * Test cases for the {@link UnivariateStatistic} class.
 * @version $Id: PercentileTest.java 1244107 2012-02-14 16:17:55Z erans $
 */
public class PercentileTest extends UnivariateStatisticAbstractTest{

    protected Percentile stat;

    /**
     * {@link org.apache.commons.math3.stat.descriptive.rank.Percentile.Type type}
     * of estimation to be used while calling {@link #getUnivariateStatistic()}
     */
    protected Type type = CM;

    /**
     * A default percentile to be used for {@link #getUnivariateStatistic()}
     */
    protected final double DEFAULT_PERCENTILE = 95d;

    /**
     * {@link org.apache.commons.math3.stat.descriptive.rank.Percentile.Type TYPE}s
     * that this test will verify against
     */
    protected static final Type[] TYPE =
            new Type[] { CM, R_1, R_2, R_3, R_4, R_5, R_6, R_7, R_8, R_9};

    /**
     * Before method to ensure defaults retained
     */
    @Before
    public void before() {
        Percentile.setPivotingStrategy(PivotingStrategy.MEDIAN_OF_3);
        type = CM;
    }

    /**
     * Gets a percentile with given percentile and given estimation type
     *
     * @param p pth Quantile to be computed
     * @param type One of the {@link Type}
     * @return Percentile object created
     */
    public UnivariateStatistic getUnivariateStatistic(double p,
            Type type) {
        return new Percentile(p, type);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public UnivariateStatistic getUnivariateStatistic() {
        return new Percentile(95.0);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double expectedValue() {
        return this.percentile95;
    }

    @Test
    public void testHighPercentile(){
        double[] d = new double[]{1, 2, 3};
        Percentile p = new Percentile(75);
        Assert.assertEquals(3.0, p.evaluate(d), 1.0e-5);
    }

    @Test
    public void testLowPercentile() {
        double[] d = new double[] {0, 1};
        Percentile p = new Percentile(25);
        Assert.assertEquals(0d, p.evaluate(d), Double.MIN_VALUE);
    }

    @Test
    public void testPercentile() {
        double[] d = new double[] {1, 3, 2, 4};
        Percentile p = new Percentile(30);
        Assert.assertEquals(1.5, p.evaluate(d), 1.0e-5);
        p.setQuantile(25);
        Assert.assertEquals(1.25, p.evaluate(d), 1.0e-5);
        p.setQuantile(75);
        Assert.assertEquals(3.75, p.evaluate(d), 1.0e-5);
        p.setQuantile(50);
        Assert.assertEquals(2.5, p.evaluate(d), 1.0e-5);

        // invalid percentiles
        try {
            p.evaluate(d, 0, d.length, -1.0);
            Assert.fail();
        } catch (MathIllegalArgumentException ex) {
            // success
        }
        try {
            p.evaluate(d, 0, d.length, 101.0);
            Assert.fail();
        } catch (MathIllegalArgumentException ex) {
            // success
        }
    }

    @Test
    public void testNISTExample() {
        double[] d = new double[] {95.1772, 95.1567, 95.1937, 95.1959,
                95.1442, 95.0610,  95.1591, 95.1195, 95.1772, 95.0925, 95.1990, 95.1682
        };
        Percentile p = new Percentile(90);
        Assert.assertEquals(95.1981, p.evaluate(d), 1.0e-4);
        Assert.assertEquals(95.1990, p.evaluate(d,0,d.length, 100d), 0);
    }

    @Test
    public void test5() {
        Percentile percentile = new Percentile(5);
        Assert.assertEquals(this.percentile5, percentile.evaluate(testArray), getTolerance());
    }

    @Test
    public void testNullEmpty() {
        Percentile percentile = new Percentile(50);
        double[] nullArray = null;
        double[] emptyArray = new double[] {};
        try {
            percentile.evaluate(nullArray);
            Assert.fail("Expecting MathIllegalArgumentException for null array");
        } catch (MathIllegalArgumentException ex) {
            // expected
        }
        Assert.assertTrue(Double.isNaN(percentile.evaluate(emptyArray)));
    }

    @Test
    public void testSingleton() {
        Percentile percentile = new Percentile(50);
        double[] singletonArray = new double[] {1d};
        Assert.assertEquals(1d, percentile.evaluate(singletonArray), 0);
        Assert.assertEquals(1d, percentile.evaluate(singletonArray, 0, 1), 0);
        Assert.assertEquals(1d, percentile.evaluate(singletonArray, 0, 1, 5), 0);
        Assert.assertEquals(1d, percentile.evaluate(singletonArray, 0, 1, 100), 0);
        Assert.assertTrue(Double.isNaN(percentile.evaluate(singletonArray, 0, 0)));
    }

    @Test
    public void testSpecialValues() {
        Percentile percentile = new Percentile(50);
        double[] specialValues = new double[] {0d, 1d, 2d, 3d, 4d,  Double.NaN};
        Assert.assertEquals(2.5d, percentile.evaluate(specialValues), 0);
        specialValues =  new double[] {Double.NEGATIVE_INFINITY, 1d, 2d, 3d,
                Double.NaN, Double.POSITIVE_INFINITY};
        Assert.assertEquals(2.5d, percentile.evaluate(specialValues), 0);
        specialValues = new double[] {1d, 1d, Double.POSITIVE_INFINITY,
                Double.POSITIVE_INFINITY};
        Assert.assertTrue(Double.isInfinite(percentile.evaluate(specialValues)));
        specialValues = new double[] {1d, 1d, Double.NaN,
                Double.NaN};
        Assert.assertTrue(Double.isNaN(percentile.evaluate(specialValues)));
        specialValues = new double[] {1d, 1d, Double.NEGATIVE_INFINITY,
                Double.NEGATIVE_INFINITY};
        // Interpolation results in NEGATIVE_INFINITY + POSITIVE_INFINITY
        Assert.assertTrue(Double.isNaN(percentile.evaluate(specialValues)));
    }

    @Test
    public void testSetQuantile() {
        Percentile percentile = new Percentile(10);
        percentile.setQuantile(100); // OK
        Assert.assertEquals(100, percentile.getQuantile(), 0);
        try {
            percentile.setQuantile(0);
            Assert.fail("Expecting MathIllegalArgumentException");
        } catch (MathIllegalArgumentException ex) {
            // expected
        }
        try {
            new Percentile(0);
            Assert.fail("Expecting MathIllegalArgumentException");
        } catch (MathIllegalArgumentException ex) {
            // expected
        }
    }

    //Below tests are basically to run for all estimation types.
    /**
     * While {@link #testHighPercentile()} checks only for the existing
     * implementation; this method verifies for all the types including CM type.
     */
    @Test
    public void testAllTechniquesHighPercentile() {
        double[] d = new double[] { 1, 2, 3 };
        testAssertMappedValues(d, new Object[][] { { CM, 3d }, { R_1, 3d },
                { R_2, 3d }, { R_3, 2d }, { R_4, 2.25 }, { R_5, 2.75 },
                { R_6, 3d }, { R_7, 2.5 },{ R_8, 2.83333 }, {R_9,2.81250} },
                75d, 1.0e-5);
    }

    @Test
    public void testAllTechniquesLowPercentile() {
        double[] d = new double[] { 0, 1 };
        testAssertMappedValues(d, new Object[][] { { CM, 0d }, { R_1, 0d },
                { R_2, 0d }, { R_3, 0d }, { R_4, 0d }, {R_5, 0d}, {R_6, 0d},
                { R_7, 0.25 }, { R_8, 0d }, {R_9, 0d} },
                25d, Double.MIN_VALUE);
    }

    @Test
    public void testAllTechniquesPercentile() {
        double[] d = new double[] { 1, 3, 2, 4 };

        testAssertMappedValues(d, new Object[][] { { CM, 1.5d },
                { R_1, 2d }, { R_2, 2d }, { R_3, 1d }, { R_4, 1.2 }, {R_5, 1.7},
                { R_6, 1.5 },{ R_7, 1.9 }, { R_8, 1.63333 },{ R_9, 1.65 } },
                30d, 1.0e-05);

        testAssertMappedValues(d, new Object[][] { { CM, 1.25d },
                { R_1, 1d }, { R_2, 1.5d }, { R_3, 1d }, { R_4, 1d }, {R_5, 1.5},
                { R_6, 1.25 },{ R_7, 1.75 },
                { R_8, 1.41667 }, { R_9, 1.43750 } }, 25d, 1.0e-05);

        testAssertMappedValues(d, new Object[][] { { CM, 3.75d },
                { R_1, 3d }, { R_2, 3.5d }, { R_3, 3d }, { R_4, 3d },
                { R_5, 3.5d },{ R_6, 3.75d }, { R_7, 3.25 },
                { R_8, 3.58333 },{ R_9, 3.56250} }, 75d, 1.0e-05);

        testAssertMappedValues(d, new Object[][] { { CM, 2.5d },
                { R_1, 2d }, { R_2, 2.5d }, { R_3, 2d }, { R_4, 2d },
                { R_5, 2.5 },{ R_6, 2.5 },{ R_7, 2.5 },
                { R_8, 2.5 },{ R_9, 2.5 } }, 50d, 1.0e-05);

        // invalid percentiles
        for (Type e : TYPE) {
            try {
                new Percentile(-1.0, e).evaluate(d,
                                0, d.length, -1.0);
                Assert.fail();
            } catch (MathIllegalArgumentException ex) {
                // success
            }
        }

        for (Type e : TYPE) {
            try {
                new Percentile(101.0, e).evaluate(d,
                        0, d.length, 101.0);
                Assert.fail();
            } catch (MathIllegalArgumentException ex) {
                // success
            }
        }
    }

    @Test
    public void testAllTechniquesPercentileUsingCentralPivoting() {
        Percentile.setPivotingStrategy(PivotingStrategy.CENTRAL);
        Assert.assertTrue(PivotingStrategy.CENTRAL
                .equals(Percentile.getPivotingStrategy()));
        testAllTechniquesPercentile();
    }

    @Test
    public void testAllTechniquesPercentileUsingRandomPivoting() {
        Percentile.setPivotingStrategy(PivotingStrategy.RANDOM);
        Assert.assertTrue(PivotingStrategy.RANDOM
                .equals(Percentile.getPivotingStrategy()));
        testAllTechniquesPercentile();
    }

    @Test
    public void testAllTechniquesNISTExample() {
        double[] d =
                new double[] { 95.1772, 95.1567, 95.1937, 95.1959, 95.1442,
                        95.0610, 95.1591, 95.1195, 95.1772, 95.0925, 95.1990,
                        95.1682 };

        testAssertMappedValues(d, new Object[][] { { CM, 95.1981 },
                { R_1, 95.19590 }, { R_2, 95.19590 }, { R_3, 95.19590 },
                { R_4, 95.19546 }, { R_5, 95.19683 }, { R_6, 95.19807 },
                { R_7, 95.19568 }, { R_8, 95.19724 }, { R_9, 95.19714 } }, 90d,
                1.0e-04);

        for (Type e : TYPE) {
            Assert.assertEquals(95.1990, getUnivariateStatistic(100d, e)
                    .evaluate(d), 1.0e-4);
        }
    }

    @Test
    public void testAllTechniques5() {
        UnivariateStatistic percentile = getUnivariateStatistic(5, CM);
        Assert.assertEquals(this.percentile5, percentile.evaluate(testArray),
                getTolerance());
        testAssertMappedValues(testArray,
                new Object[][] { { CM, percentile5 }, { R_1, 8.8000 },
                        { R_2, 8.8000 }, { R_3, 8.2000 }, { R_4, 8.2600 },
                        { R_5, 8.5600 }, { R_6, 8.2900 },
                        { R_7, 8.8100 }, { R_8, 8.4700 },
                        { R_9, 8.4925 }}, 5d, getTolerance());
    }

    @Test
    public void testAllTechniquesNullEmpty() {

        double[] nullArray = null;
        double[] emptyArray = new double[] {};
        for (Type e : TYPE) {
            UnivariateStatistic percentile = getUnivariateStatistic(50, e);
            try {
                percentile.evaluate(nullArray);
                Assert.fail("Expecting MathIllegalArgumentException "
                        + "for null array");
            } catch (MathIllegalArgumentException ex) {
                // expected
            }
            Assert.assertTrue(Double.isNaN(percentile.evaluate(emptyArray)));
        }

    }

    @Test
    public void testAllTechniquesSingleton() {
        double[] singletonArray = new double[] { 1d };
        for (Type e : TYPE) {
            UnivariateStatistic percentile = getUnivariateStatistic(50, e);
            Assert.assertEquals(1d, percentile.evaluate(singletonArray), 0);
            Assert.assertEquals(1d, percentile.evaluate(singletonArray, 0, 1),
                    0);
            Assert.assertEquals(1d,
                    new Percentile().evaluate(singletonArray, 0, 1, 5), 0);
            Assert.assertEquals(1d,
                    new Percentile().evaluate(singletonArray, 0, 1, 100), 0);
            Assert.assertTrue(Double.isNaN(percentile.evaluate(singletonArray,
                    0, 0)));
        }
    }

    @Test
    public void testAllTechniquesEmpty() {
        double[] singletonArray = new double[] { };
        for (Type e : TYPE) {
            UnivariateStatistic percentile = getUnivariateStatistic(50, e);
            Assert.assertEquals(Double.NaN, percentile.evaluate(singletonArray),
                    0);
            Assert.assertEquals(Double.NaN, percentile.evaluate(singletonArray,
                    0, 0),
                    0);
            Assert.assertEquals(Double.NaN,
                    new Percentile().evaluate(singletonArray, 0, 0, 5), 0);
            Assert.assertEquals(Double.NaN,
                    new Percentile().evaluate(singletonArray, 0, 0, 100), 0);
            Assert.assertTrue(Double.isNaN(percentile.evaluate(singletonArray,
                    0, 0)));
        }
    }

    @Test(expected=NullArgumentException.class)
    public void testSetNullPivotingStrategy() {
        Percentile.setPivotingStrategy(null);
    }


    @Test
    public void testReplaceNanInRange() {
        double[] specialValues =
                new double[] { 0d, 1d, 2d, 3d, 4d, Double.NaN, Double.NaN, 5d,
                7d, Double.NaN, 8d};
        Assert.assertEquals(5d,new Percentile(50d).evaluate(specialValues),0d);
        Assert.assertEquals(3d,new Percentile(50d,Type.R_1)
                                                .evaluate(specialValues),0d);
        Assert.assertEquals(3.5d,new Percentile(50d,Type.R_2)
                                                .evaluate(specialValues),0d);

    }

    @Test
    public void testRemoveNan() {
        double[] specialValues =
                new double[] { 0d, 1d, 2d, 3d, 4d, Double.NaN };
        double[] expectedValues =
                new double[] { 0d, 1d, 2d, 3d, 4d };
        Assert.assertEquals(2.0,
                new Percentile(50d,Type.R_1).evaluate(specialValues),0d);
        Assert.assertEquals(2.0,
                new Percentile(50d,Type.R_1).evaluate(expectedValues),0d);
        Assert.assertTrue(Double.isNaN(
                new Percentile(50d,Type.R_1).evaluate(specialValues,5,1)));
        Assert.assertEquals(4d,new Percentile(50d,Type.R_1)
                                        .evaluate(specialValues,4,2),0d);
        Assert.assertEquals(3d,new Percentile(50d,Type.R_1)
        .evaluate(specialValues,3,3),0d);
        Assert.assertEquals(3.5d,new Percentile(50d,Type.R_2)
        .evaluate(specialValues,3,3),0d);

    }

    @Test
    public void testPercentileCopy() {
       Percentile original= new Percentile(50d,Type.CM);
       Percentile copy = new Percentile(original);
       Assert.assertEquals(original.getNaNStrategy(),copy.getNaNStrategy());
       Assert.assertEquals(original.getQuantile(), copy.getQuantile(),0d);
       Assert.assertEquals(original.getEstimationType(),copy.getEstimationType());
       Assert.assertEquals(NaNStrategy.FIXED, original.getNaNStrategy());
    }

    @Test
    public void testAllTechniquesSpecialValues() {
        UnivariateStatistic percentile = getUnivariateStatistic(50d, CM);
        double[] specialValues =
                new double[] { 0d, 1d, 2d, 3d, 4d, Double.NaN };
        Assert.assertEquals(2.5d, percentile.evaluate(specialValues), 0);

        testAssertMappedValues(specialValues, new Object[][] {
                { CM, 2.5d }, { R_1, 2.0 }, { R_2, 2.0 }, { R_3, 1.0 },
                { R_4, 1.5 }, { R_5, 2.0 }, { R_6, 2.0 },
                { R_7, 2.0 }, { R_8, 2.0 }, { R_9, 2.0 }}, 50d, 0d);

        specialValues =
                new double[] { Double.NEGATIVE_INFINITY, 1d, 2d, 3d,
                        Double.NaN, Double.POSITIVE_INFINITY };
        Assert.assertEquals(2.5d, percentile.evaluate(specialValues), 0);

        testAssertMappedValues(specialValues, new Object[][] {
                { CM, 2.5d }, { R_1, 2.0 }, { R_2, 2.0 }, { R_3, 1.0 },
                { R_4, 1.5 }, { R_5, 2.0 }, { R_7, 2.0 }, { R_7, 2.0 },
                { R_8, 2.0 }, { R_9, 2.0 } }, 50d, 0d);

        specialValues =
                new double[] { 1d, 1d, Double.POSITIVE_INFINITY,
                        Double.POSITIVE_INFINITY };
        Assert.assertTrue(Double.isInfinite(percentile.evaluate(specialValues)));

        testAssertMappedValues(specialValues, new Object[][] {
                // This is one test not matching with R results.
                { CM, Double.POSITIVE_INFINITY },
                { R_1,/* 1.0 */Double.NaN },
                { R_2, /* Double.POSITIVE_INFINITY */Double.NaN },
                { R_3, /* 1.0 */Double.NaN }, { R_4, /* 1.0 */Double.NaN },
                { R_5, Double.POSITIVE_INFINITY },
                { R_6, Double.POSITIVE_INFINITY },
                { R_7, Double.POSITIVE_INFINITY },
                { R_8, Double.POSITIVE_INFINITY },
                { R_9, Double.POSITIVE_INFINITY }, }, 50d, 0d);

        specialValues = new double[] { 1d, 1d, Double.NaN, Double.NaN };
        Assert.assertTrue(Double.isNaN(percentile.evaluate(specialValues)));
        testAssertMappedValues(specialValues, new Object[][] {
                { CM, Double.NaN }, { R_1, 1.0 }, { R_2, 1.0 }, { R_3, 1.0 },
                { R_4, 1.0 }, { R_5, 1.0 },{ R_6, 1.0 },{ R_7, 1.0 },
                { R_8, 1.0 }, { R_9, 1.0 },}, 50d, 0d);

        specialValues =
                new double[] { 1d, 1d, Double.NEGATIVE_INFINITY,
                        Double.NEGATIVE_INFINITY };

        testAssertMappedValues(specialValues, new Object[][] {
                { CM, Double.NaN }, { R_1, Double.NaN },
                { R_2, Double.NaN }, { R_3, Double.NaN }, { R_4, Double.NaN },
                { R_5, Double.NaN }, { R_6, Double.NaN },
                { R_7, Double.NaN }, { R_8, Double.NaN }, { R_9, Double.NaN }
                }, 50d, 0d);

    }

    @Test
    public void testAllTechniquesSetQuantile() {
        for (Type e : TYPE) {
            Percentile percentile = new Percentile(10, e);
            percentile.setQuantile(100); // OK
            Assert.assertEquals(100, percentile.getQuantile(), 0);
            try {
                percentile.setQuantile(0);
                Assert.fail("Expecting MathIllegalArgumentException");
            } catch (MathIllegalArgumentException ex) {
                // expected
            }
            try {
                new Percentile(0);
                Assert.fail("Expecting MathIllegalArgumentException");
            } catch (MathIllegalArgumentException ex) {
                // expected
            }
        }
    }

    @Test
    public void testAllTechniquesEvaluateArraySegmentWeighted() {
        for (Type e : TYPE) {
            type = e;
            testEvaluateArraySegmentWeighted();
        }
    }

    @Test
    public void testAllTechniquesEvaluateArraySegment() {
        for (Type e : TYPE) {
            type = e;
            testEvaluateArraySegment();
        }
    }


    @Test
    public void testAllTechniquesWeightedConsistency() {
        for (Type e : TYPE) {
            type = e;
            testWeightedConsistency();
        }
    }

    @Test
    public void testAllTechniquesCopy() {
        for (Type e : TYPE) {
            type = e;
            testCopy();
        }
    }

    @Test
    public void testAllTechniquesEvaluation() {

        testAssertMappedValues(testArray, new Object[][] { { CM, 20.820 },
                { R_1, 19.800 }, { R_2, 19.800 }, { R_3, 19.800 },
                { R_4, 19.310 }, { R_5, 20.280 }, { R_6, 20.820 },
                { R_7, 19.555 }, { R_8, 20.460 },{ R_9, 20.415} },
                DEFAULT_PERCENTILE, tolerance);
    }

    @Test
    public void testPercentileWithTechnique() {
        Percentile p = new Percentile(CM);
        Assert.assertTrue(CM.equals(p.getEstimationType()));
        Assert.assertFalse(R_1.equals(p.getEstimationType()));
    }

    static final int TINY = 10, SMALL = 50, NOMINAL = 100, MEDIUM = 500,
            STANDARD = 1000, BIG = 10000, VERY_BIG = 50000, LARGE = 1000000,
            VERY_LARGE = 10000000;
    static final int[] sampleSizes= {TINY , SMALL , NOMINAL , MEDIUM ,
            STANDARD, BIG };

    @Test
    public void testStoredVsDirect() {
        RandomGenerator rand= new JDKRandomGenerator();
        rand.setSeed(Long.MAX_VALUE);
        for(int sampleSize:sampleSizes) {
            double[] data = new NormalDistribution(rand,4000, 50)
                                .sample(sampleSize);
            for(double p:new double[] {50d,95d}) {
                for(Type e:Type.values()) {
                    Percentile pStoredData = new Percentile(p,e);
                    pStoredData.setData(data);
                    double storedDataResult=pStoredData.evaluate();
                    pStoredData.setData(null);
                    Percentile pDirect = new Percentile(p,e);
                    Assert.assertEquals("Sample="+sampleSize+",P="+p+" e="+e,
                            storedDataResult,
                            pDirect.evaluate(data),0d);
                }
            }
        }
    }
    @Test
    public void testPercentileWithDataRef() {
        Percentile p = new Percentile(R_7);
        p.setData(testArray);
        Assert.assertTrue(R_7.equals(p.getEstimationType()));
        Assert.assertFalse(R_1.equals(p.getEstimationType()));
        Assert.assertEquals(12d, p.evaluate(), 0d);
        Assert.assertEquals(12.16d, p.evaluate(60d), 0d);
    }

    @SuppressWarnings("deprecation")
    @Test(expected=MathUnsupportedOperationException.class)
    public void testMedianOf3() {
        Percentile p = new Percentile(R_7);
        Assert.assertEquals(0, p.medianOf3(testArray, 0, testArray.length));
        Assert.assertEquals(10,
                p.medianOf3(testWeightsArray, 0, testWeightsArray.length));
    }

    @Test(expected=NullArgumentException.class)
    public void testNullEstimation() {
        //Check if null estimation type can be injected
        Assert.assertNull(new Percentile(
                (Type)null).getEstimationType());
        Assert.fail("Unexpected: Percentile cannot have a NULL " +
                "Estimation type");
    }

    @Test
    public void testAllEstimationTechniquesOnlyLimits() {
        final int N=testArray.length;

        double[] input=MathArrays.copyOf(testArray);
        Arrays.sort(input);
        double min = input[0];
        double max=input[input.length-1];
        //limits may be ducked by 0.01 to induce the condition of p<pMin
        Object[][] map =
                new Object[][] { { CM, 0d, 1d }, { R_1, 0d, 1d },
                        { R_2, 0d,1d }, { R_3, 0.5/N,1d },
                        { R_4, 1d/N-0.001,1d },
                        { R_5, 0.5/N-0.001,(N-0.5)/N}, { R_6, 0.99d/(N+1),
                            1.01d*N/(N+1)},
                        { R_7, 0d,1d}, { R_8, 1.99d/3/(N+1d/3),
                            (N-1d/3)/(N+1d/3)},
                        { R_9, 4.99d/8/(N+0.25), (N-3d/8)/(N+0.25)} };

        for(Object[] arr:map) {
            Type t=(Type)arr[0];
            double pMin=(Double)arr[1];
            double pMax=(Double)arr[2];
            Assert.assertEquals("Type:"+t,0d, t.index(pMin, N),0d);
            Assert.assertEquals("Type:"+t,N, t.index(pMax, N),0.5d);
            pMin=pMin==0d?pMin+0.01:pMin;
            testAssertMappedValues(testArray, new Object[][] { { t, min }}
                ,pMin, 0.01);

            testAssertMappedValues(testArray, new Object[][] { { t, max }}
            ,pMax*100, tolerance);
        }
    }

    @Test
    public void testAllEstimationTechniquesOnly() {
        Assert.assertEquals("Commons Math",CM.getName());
        Object[][] map =
                new Object[][] { { CM, 20.82 }, { R_1, 19.8 },
                        { R_2, 19.8 }, { R_3, 19.8 }, { R_4, 19.310 },
                        { R_5, 20.280}, { R_6, 20.820},
                        { R_7, 19.555 }, { R_8, 20.460 },{R_9,20.415} };
        try {
            CM.evaluate(testArray,
                    -1d );
        }catch(OutOfRangeException oore) {}
        try {
            CM.evaluate(testArray,
                    101d );
        }catch(OutOfRangeException oore) {}
        try {
            CM.evaluate(testArray,
                    50d );
        }catch(OutOfRangeException oore) {}
        for (Object[] o : map) {
            Type e = (Type) o[0];
            double expected = (Double) o[1];
            double result =
                    e.evaluate(testArray, DEFAULT_PERCENTILE );
            Assert.assertEquals("expected[" + e + "] = " + expected +
                    " but was = " + result, expected, result, tolerance);
        }
    }
    @Test
    public void testAllEstimationTechniquesOnlyForAllPivotingStrategies() {

        Assert.assertEquals("Commons Math",CM.getName());

        for(PivotingStrategy strategy:PivotingStrategy.values()) {
            Percentile.setPivotingStrategy(strategy);
            testAllEstimationTechniquesOnly();
        }
    }


    @Test
    public void testAllEstimationTechniquesOnlyForExtremeIndexes() {
        final double MAX=100;
        Object[][] map =
                new Object[][] { { CM, 0d, MAX}, { R_1, 0d,MAX+0.5 },
                { R_2, 0d,MAX}, { R_3, 0d,MAX }, { R_4, 0d,MAX },
                { R_5, 0d,MAX }, { R_6, 0d,MAX },
                { R_7, 0d,MAX }, { R_8, 0d,MAX }, { R_9, 0d,MAX }  };
        for (Object[] o : map) {
            Type e = (Type) o[0];
                Assert.assertEquals(((Double)o[1]).doubleValue(),
                        e.index(0d, (int)MAX),0d);
                Assert.assertEquals("Enum:"+e,((Double)o[2]).doubleValue(),
                        e.index(1.0, (int)MAX),0d);
            }
    }
    @Test
    public void testAllEstimationTechniquesOnlyForNullsAndOOR() {

        Object[][] map =
                new Object[][] { { CM, 20.82 }, { R_1, 19.8 },
                        { R_2, 19.8 }, { R_3, 19.8 }, { R_4, 19.310 },
                        { R_5, 20.280}, { R_6, 20.820},
                        { R_7, 19.555 }, { R_8, 20.460 },{ R_9, 20.415 } };
        for (Object[] o : map) {
            Type e = (Type) o[0];
            try {

                e.evaluate(null, DEFAULT_PERCENTILE );
                Assert.fail("Expecting NullArgumentException");
            } catch (NullArgumentException nae) {
                // expected
            }
            try {
                e.evaluate(testArray, 120);
                Assert.fail("Expecting OutOfRangeException");
            } catch (OutOfRangeException oore) {
                // expected
            }
        }
    }

    /**
     * Simple test assertion utility method assuming {@link NaNStrategy default}
     * nan handling strategy specific to each {@link Type type}
     *
     * @param data input data
     * @param map of expected result against a {@link Type}
     * @param p the quantile to compute for
     * @param tolerance the tolerance of difference allowed
     */
    protected void testAssertMappedValues(double[] data, Object[][] map,
            Double p, Double tolerance) {
        for (Object[] o : map) {
            Type e = (Type) o[0];
            double expected = (Double) o[1];
            try {
                double result = getUnivariateStatistic(p, e).evaluate(data);
                Assert.assertEquals("expected[" + e + "] = " + expected +
                    " but was = " + result, expected, result, tolerance);
            }catch(Exception ex) {
                Assert.fail("Exception occured for estimation type "+e+":"+
                        ex.getLocalizedMessage());
            }
        }
    }

    /**A Simple extension to show overriding of default NaN Strategy per type.*/
    @SuppressWarnings("serial")
    private static class NewPercentile extends Percentile{
        /** map overrides the default mapping of type to NaN Strategy*/
        private static final Map<Type,NaNStrategy> map =
                new HashMap<Type,NaNStrategy>() {
            {
            putAll(Percentile.DEFAULT_NAN_HANDLING_PER_TYPE);
            put(CM, NaNStrategy.MAXIMAL);
            put(R_7, NaNStrategy.MINIMAL);
            put(R_9, NaNStrategy.FAILED);
            }
        };
        /**
         * Constructor
         * @param p the pth quantile
         * @param type an instance of {@link Type}
         */
        public NewPercentile(double p, Type type) {
            super(p, type, map.get(type)); //invokes the protected constructor
        }
    }

    // Some NaNStrategy specific testing
    @Test(expected=NotANumberException.class)
    public void testNanStrategyFailed() {
        double[] specialValues =
                new double[] { 0d, 1d, 2d, 3d, 4d, Double.NaN };
        Assert.assertTrue(Double.isNaN(new NewPercentile(50d,Type.CM)
        .evaluate(specialValues,3,3)));
        Assert.assertEquals(2d,new NewPercentile(50d,Type.R_1)
        .evaluate(specialValues),0d);
        Assert.assertEquals(Double.NaN,new NewPercentile(50d,Type.R_5)
        .evaluate(new double[] {Double.NaN,Double.NaN,Double.NaN}),0d);
        Assert.assertEquals(50d,new NewPercentile(50d,Type.R_7).evaluate(
                new double[] {50d,50d,50d},1,2),0d);
        new NewPercentile(50d,Type.R_9).evaluate(specialValues, 3, 3);
    }

    @Test
    public void testAllTechniquesSpecialValuesWithNaNStrategy() {
        double[] specialValues =
                new double[] { 0d, 1d, 2d, 3d, 4d, Double.NaN };
        try {
            new Percentile(50d,Type.CM,null);
            Assert.fail("Expecting NullArgumentArgumentException "
                    + "for null Nan Strategy");
        } catch (NullArgumentException ex) {
            // expected
        }
        //This is as per each type's default NaNStrategy
        testAssertMappedValues(specialValues, new Object[][] {
                { CM, 2.5d }, { R_1, 2.0 }, { R_2, 2.0 }, { R_3, 1.0 },
                { R_4, 1.5 }, { R_5, 2.0 }, { R_6, 2.0 },
                { R_7, 2.0 }, { R_8, 2.0 }, { R_9, 2.0 }}, 50d, 0d);

        //This is as per MAXIMAL and hence the values tend a +0.5 upward
        testAssertMappedValues(specialValues, new Object[][] {
                { CM, 2.5d }, { R_1, 2.0 }, { R_2, 2.5 }, { R_3, 2.0 },
                { R_4, 2.0 }, { R_5, 2.5 }, { R_6, 2.5 },
                { R_7, 2.5 }, { R_8, 2.5 }, { R_9, 2.5 }}, 50d, 0d,
                NaNStrategy.MAXIMAL);

        //This is as per MINIMAL and hence the values tend a -0.5 downward
        testAssertMappedValues(specialValues, new Object[][] {
                { CM, 1.5d }, { R_1, 1.0 }, { R_2, 1.5 }, { R_3, 1.0 },
                { R_4, 1.0 }, { R_5, 1.5 }, { R_6, 1.5 },
                { R_7, 1.5 }, { R_8, 1.5 }, { R_9, 1.5 }}, 50d, 0d,
                NaNStrategy.MINIMAL);

        //This is as per REMOVED as here CM changed its value from default
        //while rest of Estimation types were anyways defaulted to REMOVED
        testAssertMappedValues(specialValues, new Object[][] {
                { CM, 2.0 }, { R_1, 2.0 }, { R_2, 2.0 }, { R_3, 1.0 },
                { R_4, 1.5 }, { R_5, 2.0 }, { R_6, 2.0 },
                { R_7, 2.0 }, { R_8, 2.0 }, { R_9, 2.0 }}, 50d, 0d,
                NaNStrategy.REMOVED);
    }

    /**
     * Simple test assertion utility method
     *
     * @param data input data
     * @param map of expected result against a {@link Type}
     * @param p the quantile to compute for
     * @param tolerance the tolerance of difference allowed
     * @param nanStrategy NaNStrategy to be passed
     */
    protected void testAssertMappedValues(double[] data, Object[][] map,
            Double p, Double tolerance, NaNStrategy nanStrategy) {
        for (Object[] o : map) {
            Type e = (Type) o[0];
            double expected = (Double) o[1];
            try {
                double result = new Percentile(p, e, nanStrategy).evaluate(data);
                Assert.assertEquals("expected[" + e + "] = " + expected +
                    " but was = " + result, expected, result, tolerance);
            }catch(Exception ex) {
                Assert.fail("Exception occured for estimation type "+e+":"+
                        ex.getLocalizedMessage());
            }
        }
    }
}
