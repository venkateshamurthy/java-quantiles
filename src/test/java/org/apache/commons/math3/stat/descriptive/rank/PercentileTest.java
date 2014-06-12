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
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.commons.math3.exception.MathIllegalArgumentException;
import org.apache.commons.math3.exception.MathUnsupportedOperationException;
import org.apache.commons.math3.exception.NullArgumentException;
import org.apache.commons.math3.exception.OutOfRangeException;
import org.apache.commons.math3.stat.descriptive.UnivariateStatistic;
import org.apache.commons.math3.stat.descriptive.UnivariateStatisticAbstractTest;
import org.apache.commons.math3.stat.descriptive.rank.Percentile.PivotingStrategy;
import org.apache.commons.math3.stat.descriptive.rank.Percentile.Type;
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
    /**
     * vmurthy changes
     * The below tests are done for all the estimation types as it
     * basically picks up the same tests specified in the enclosing class but
     * applies to all types as elucidated in {@link EstimationTechnique}.
     */
    /**
     * type of estimation to be used while calling
     * {@link #getUnivariateStatistic()}
     */
    protected Type type = CM;

    /**
     * A default percentile to be used for {@link #getUnivariateStatistic()}
     */
    protected final double DEFAULT_PERCENTILE = 95d;

    /**
     * {@link Type}s that this test will verify against
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

    @Test
    public void testAllTechniquesHighPercentile() {
        double[] d = new double[] { 1, 2, 3 };
        testAssertMappedValues(d, new Object[][] { { CM, 3d }, { R_1, 3d },
                { R_2, 3d }, { R_3, 2d }, { R_4, 2.25 }, {R_5,2.75}, {R_6,3.0},
                { R_7, 2.5 },{ R_8, 2.83333 }, {R_9,2.81250} }, 75d, 1.0e-5);
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
                new Percentile(-1.0, e).evaluate(d, 0, d.length, -1.0);
                Assert.fail();
            } catch (MathIllegalArgumentException ex) {
                // success
            }
        }

        for (Type e : TYPE) {
            try {
                new Percentile(101.0, e).evaluate(d, 0, d.length, 101.0);
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
        Assert.assertTrue(specialValues==CM.preProcess(specialValues,
                new AtomicInteger(specialValues.length)));
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


            try {
                e.preProcess(testArray, null);
            }catch(MathIllegalArgumentException ex) {
                //expected
            }
            try {
                e.preProcess(testArray, new AtomicInteger(-1));
            }catch(MathIllegalArgumentException ex) {
                //expected
            }
            try {
                e.preProcess(null, new AtomicInteger(10));
            }catch(MathIllegalArgumentException ex) {
                //expected
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
            CM.evaluate(testArray, testArray.length,
                    -1d );
        }catch(OutOfRangeException oore) {}
        try {
            CM.evaluate(testArray, testArray.length,
                    101d );
        }catch(OutOfRangeException oore) {}
        try {
            CM.evaluate(testArray, -1,
                    50d );
        }catch(OutOfRangeException oore) {}
        try {
            CM.evaluate(testArray, testArray.length+1,
                    50d );
        }catch(OutOfRangeException oore) {}
        for (Object[] o : map) {
            Type e = (Type) o[0];
            double expected = (Double) o[1];
            double result =
                    e.evaluate(testArray, testArray.length,
                            DEFAULT_PERCENTILE );
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
                        { R_7, 19.555 }, { R_8, 20.460 },{R_9,20.415} };
        for (Object[] o : map) {
            Type e = (Type) o[0];
            try {

                e.evaluate(null, testArray.length, DEFAULT_PERCENTILE );
                Assert.fail("Expecting NullArgumentException");
            } catch (NullArgumentException nae) {
                // expected
            }
            try {
                e.evaluate(testArray, -1, DEFAULT_PERCENTILE );
                Assert.fail("Expecting OutOfRangeException");
            } catch (OutOfRangeException oore) {
                // expected
            }
            try {
                e.evaluate(testArray, testArray.length, 120);
                Assert.fail("Expecting OutOfRangeException");
            } catch (OutOfRangeException oore) {
                // expected
            }
        }
    }

    /**
     * Simple test assertion utility method
     *
     * @param d input data
     * @param map of expected result against a {@link Type}
     * @param p the quantile to compute for
     * @param tolerance the tolerance of difference allowed
     */
    protected void testAssertMappedValues(double[] d, Object[][] map, Double p,
            Double tolerance) {
        for (Object[] o : map) {
            Type e = (Type) o[0];
            double expected = (Double) o[1];
            double result = getUnivariateStatistic(p, e).evaluate(d);
            Assert.assertEquals("expected[" + e + "] = " + expected +
                    " but was = " + result, expected, result, tolerance);
        }
    }

}
