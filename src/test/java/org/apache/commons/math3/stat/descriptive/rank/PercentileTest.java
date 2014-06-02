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
import static org.apache.commons.math3.stat.descriptive.rank.Percentile.EstimationTechnique.DEFAULT;
import static org.apache.commons.math3.stat.descriptive.rank.Percentile.EstimationTechnique.R1;
import static org.apache.commons.math3.stat.descriptive.rank.Percentile.EstimationTechnique.R2;
import static org.apache.commons.math3.stat.descriptive.rank.Percentile.EstimationTechnique.R3;
import static org.apache.commons.math3.stat.descriptive.rank.Percentile.EstimationTechnique.R4;
import static org.apache.commons.math3.stat.descriptive.rank.Percentile.EstimationTechnique.R7;
import static org.apache.commons.math3.stat.descriptive.rank.Percentile.EstimationTechnique.R8;

import org.apache.commons.math3.exception.MathIllegalArgumentException;
import org.apache.commons.math3.exception.NullArgumentException;
import org.apache.commons.math3.exception.OutOfRangeException;
import org.apache.commons.math3.stat.descriptive.UnivariateStatistic;
import org.apache.commons.math3.stat.descriptive.UnivariateStatisticAbstractTest;
import org.apache.commons.math3.stat.descriptive.rank.Percentile.EstimationTechnique;
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
     * The below tests are done for all the estimation techniques as it
     * basically picks up the same tests specified in the enclosing class but
     * applies to all techniques as elucidated in {@link EstimationTechnique}.
     */
    /**
     * estimationTechnique to be used while calling
     * {@link #getUnivariateStatistic()}
     */
    protected EstimationTechnique estimationTechnique = DEFAULT;

    /**
     * A default percentile to be used for {@link #getUnivariateStatistic()}
     */
    protected final double DEFAULT_PERCENTILE = 95d;

    /**
     * {@link EstimationTechnique}s that this test will verify against
     */
    protected final EstimationTechnique[] ESTIMATION_TECHNIQUES =
            new EstimationTechnique[] { DEFAULT, R1, R2, R3, R4, R7, R8 };

    /**
     * Before method to ensure defaults retained
     */
    @Before
    public void before() {
        estimationTechnique = DEFAULT;
    }

    /**
     * Gets a percentile with given percentile and given estimation technique
     *
     * @param p pth Quantile to be computed
     * @param technique One of the {@link EstimationTechnique}
     * @return Percentile object created
     */
    public UnivariateStatistic getUnivariateStatistic(double p,
            EstimationTechnique technique) {
        return new Percentile(p, technique);
    }

    @Test
    public void testAllTechniquesHighPercentile() {
        double[] d = new double[] { 1, 2, 3 };
        testAssertMappedValues(d, new Object[][] { { DEFAULT, 3d }, { R1, 3d },
                { R2, 3d }, { R3, 2d }, { R4, 2.25 }, { R7, 2.5 },
                { R8, 2.83333 } }, 75d, 1.0e-5);
    }

    @Test
    public void testAllTechniquesLowPercentile() {
        double[] d = new double[] { 0, 1 };
        testAssertMappedValues(d, new Object[][] { { DEFAULT, 0d }, { R1, 0d },
                { R2, 0d }, { R3, 0d }, { R4, 0d }, { R7, 0.25 }, { R8, 0d } },
                25d, Double.MIN_VALUE);
    }

    @Test
    public void testAllTechniquesPercentile() {
        double[] d = new double[] { 1, 3, 2, 4 };

        testAssertMappedValues(d, new Object[][] { { DEFAULT, 1.5d },
                { R1, 2d }, { R2, 2d }, { R3, 1d }, { R4, 1.2d }, { R7, 1.9 },
                { R8, 1.63333 } }, 30d, 1.0e-05);

        testAssertMappedValues(d, new Object[][] { { DEFAULT, 1.25d },
                { R1, 1d }, { R2, 1.5d }, { R3, 1d }, { R4, 1d }, { R7, 1.75 },
                { R8, 1.41667 }, }, 25d, 1.0e-05);

        testAssertMappedValues(d, new Object[][] { { DEFAULT, 3.75d },
                { R1, 3d }, { R2, 3.5d }, { R3, 3d }, { R4, 3d }, { R7, 3.25 },
                { R8, 3.58333 }, }, 75d, 1.0e-05);

        testAssertMappedValues(d, new Object[][] { { DEFAULT, 2.5d },
                { R1, 2d }, { R2, 2.5d }, { R3, 2d }, { R4, 2d }, { R7, 2.5 },
                { R8, 2.5 }, }, 50d, 1.0e-05);

        // invalid percentiles
        for (EstimationTechnique e : ESTIMATION_TECHNIQUES) {
            try {
                new Percentile(-1.0, e).evaluate(d, 0, d.length, -1.0);
                Assert.fail();
            } catch (MathIllegalArgumentException ex) {
                // success
            }
        }

        for (EstimationTechnique e : ESTIMATION_TECHNIQUES) {
            try {
                new Percentile(101.0, e).evaluate(d, 0, d.length, 101.0);
                Assert.fail();
            } catch (MathIllegalArgumentException ex) {
                // success
            }
        }
    }

    @Test
    public void testAllTechniquesNISTExample() {
        double[] d =
                new double[] { 95.1772, 95.1567, 95.1937, 95.1959, 95.1442,
                        95.0610, 95.1591, 95.1195, 95.1772, 95.0925, 95.1990,
                        95.1682 };

        testAssertMappedValues(d, new Object[][] { { DEFAULT, 95.1981 },
                { R1, 95.19590 }, { R2, 95.19590 }, { R3, 95.19590 },
                { R4, 95.19546 }, { R7, 95.19568 }, { R8, 95.19724 }, }, 90d,
                1.0e-04);

        for (EstimationTechnique e : ESTIMATION_TECHNIQUES) {
            Assert.assertEquals(95.1990, getUnivariateStatistic(100d, e)
                    .evaluate(d), 1.0e-4);
        }
    }

    @Test
    public void testAllTechniques5() {
        UnivariateStatistic percentile = getUnivariateStatistic(5, DEFAULT);
        Assert.assertEquals(this.percentile5, percentile.evaluate(testArray),
                getTolerance());
        testAssertMappedValues(testArray,
                new Object[][] { { DEFAULT, percentile5 }, { R1, 8.8000 },
                        { R2, 8.8000 }, { R3, 8.2000 }, { R4, 8.2600 },
                        { R7, 8.8100 }, { R8, 8.4700 }, }, 5d, getTolerance());
    }

    @Test
    public void testAllTechniquesNullEmpty() {

        double[] nullArray = null;
        double[] emptyArray = new double[] {};
        for (EstimationTechnique e : ESTIMATION_TECHNIQUES) {
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
        for (EstimationTechnique e : ESTIMATION_TECHNIQUES) {
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
    public void testAllTechniquesSpecialValues() {
        UnivariateStatistic percentile = getUnivariateStatistic(50d, DEFAULT);
        double[] specialValues =
                new double[] { 0d, 1d, 2d, 3d, 4d, Double.NaN };
        Assert.assertEquals(2.5d, percentile.evaluate(specialValues), 0);

        testAssertMappedValues(specialValues, new Object[][] {
                { DEFAULT, 2.5d }, { R1, 2.0 }, { R2, 2.0 }, { R3, 1.0 },
                { R4, 1.5 }, { R7, 2.0 }, { R8, 2.0 }, }, 50d, 0d);

        specialValues =
                new double[] { Double.NEGATIVE_INFINITY, 1d, 2d, 3d,
                        Double.NaN, Double.POSITIVE_INFINITY };
        Assert.assertEquals(2.5d, percentile.evaluate(specialValues), 0);

        testAssertMappedValues(specialValues, new Object[][] {
                { DEFAULT, 2.5d }, { R1, 2.0 }, { R2, 2.0 }, { R3, 1.0 },
                { R4, 1.5 }, { R7, 2.0 }, { R8, 2.0 }, }, 50d, 0d);

        specialValues =
                new double[] { 1d, 1d, Double.POSITIVE_INFINITY,
                        Double.POSITIVE_INFINITY };
        Assert.assertTrue(Double.isInfinite(percentile.evaluate(specialValues)));

        testAssertMappedValues(specialValues, new Object[][] {
                // This is one test not matching with R results.
                { DEFAULT, Double.POSITIVE_INFINITY },
                { R1,/* 1.0 */Double.NaN },
                { R2, /* Double.POSITIVE_INFINITY */Double.NaN },
                { R3, /* 1.0 */Double.NaN }, { R4, /* 1.0 */Double.NaN },
                { R7, Double.POSITIVE_INFINITY },
                { R8, Double.POSITIVE_INFINITY }, }, 50d, 0d);

        specialValues = new double[] { 1d, 1d, Double.NaN, Double.NaN };
        Assert.assertTrue(Double.isNaN(percentile.evaluate(specialValues)));

        testAssertMappedValues(specialValues, new Object[][] {
                { DEFAULT, Double.NaN }, { R1, 1.0 }, { R2, 1.0 }, { R3, 1.0 },
                { R4, 1.0 }, { R7, 1.0 }, { R8, 1.0 }, }, 50d, 0d);

        specialValues =
                new double[] { 1d, 1d, Double.NEGATIVE_INFINITY,
                        Double.NEGATIVE_INFINITY };

        testAssertMappedValues(specialValues, new Object[][] {
                { DEFAULT, Double.NaN }, { R1, Double.NaN },
                { R2, Double.NaN }, { R3, Double.NaN }, { R4, Double.NaN },
                { R7, Double.NaN }, { R8, Double.NaN }, }, 50d, 0d);

    }

    @Test
    public void testAllTechniquesSetQuantile() {
        for (EstimationTechnique e : ESTIMATION_TECHNIQUES) {
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
        for (EstimationTechnique e : ESTIMATION_TECHNIQUES) {
            estimationTechnique = e;
            testEvaluateArraySegmentWeighted();
        }
    }

    @Test
    public void testAllTechniquesEvaluateArraySegment() {
        for (EstimationTechnique e : ESTIMATION_TECHNIQUES) {
            estimationTechnique = e;
            testEvaluateArraySegment();
        }
    }

    @Test
    public void testAllTechniquesWeightedConsistency() {
        for (EstimationTechnique e : ESTIMATION_TECHNIQUES) {
            estimationTechnique = e;
            testWeightedConsistency();
        }
    }

    @Test
    public void testAllTechniquesCopy() {
        for (EstimationTechnique e : ESTIMATION_TECHNIQUES) {
            estimationTechnique = e;
            testCopy();
        }
    }

    @Test
    public void testAllTechniquesEvaluation() {

        testAssertMappedValues(testArray, new Object[][] { { DEFAULT, 20.82 },
                { R1, 19.8 }, { R2, 19.8 }, { R3, 19.8 }, { R4, 19.310 },
                { R7, 19.555 }, { R8, 20.460 } }, DEFAULT_PERCENTILE, tolerance);
    }

    @Test
    public void testPercentileWithTechnique() {
        Percentile p = new Percentile(DEFAULT);
        Assert.assertTrue(DEFAULT.equals(p.getEstimationTechnique()));
        Assert.assertFalse(R1.equals(p.getEstimationTechnique()));
    }

    @Test
    public void testPercentileWithDataRef() {
        Percentile p = new Percentile(R7);
        p.setData(testArray);
        Assert.assertTrue(R7.equals(p.getEstimationTechnique()));
        Assert.assertFalse(R1.equals(p.getEstimationTechnique()));
        Assert.assertEquals(12d, p.evaluate(), 0d);
        Assert.assertEquals(12.16d, p.evaluate(60d), 0d);

    }

    @SuppressWarnings("deprecation")
    @Test
    public void testMedianOf3() {
        Percentile p = new Percentile(R7);
        Assert.assertEquals(0, p.medianOf3(testArray, 0, testArray.length));
        Assert.assertEquals(10,
                p.medianOf3(testWeightsArray, 0, testWeightsArray.length));
    }

    @SuppressWarnings("deprecation")
    @Test
    public void testMedianOf3WithOutOfRangeIndexes() {
        Percentile p = new Percentile(R7);
        Assert.assertEquals(0, p.medianOf3(testArray, 0, testArray.length));
        try {
            Assert.assertEquals(10,
                    p.medianOf3(testWeightsArray, -1, testWeightsArray.length));
            Assert.fail("Unexpected : Out of range begin index(-1) "
                    + "should not be accepted");
        } catch (OutOfRangeException oore) {
            // expected
        }
        try {
            Assert.assertEquals(10, p.medianOf3(testWeightsArray,
                    testWeightsArray.length + 10, testWeightsArray.length));
            Assert.fail("Unexpected : Out of range begin index(ArrayLength+10)"
                    + " should not be accepted");
        } catch (OutOfRangeException oore) {
            // expected
        }
        try {
            Assert.assertEquals(10, p.medianOf3(testWeightsArray, 0, -1));
            Assert.fail("Unexpected : Out of range end index(<begin) "
                    + "should not be accepted");
        } catch (OutOfRangeException oore) {
            // expected
        }
        try {
            Assert.assertEquals(10, p.medianOf3(testWeightsArray, 0,
                    testWeightsArray.length + 10));
            Assert.fail("Unexpected : Out of range end index(ArrayLength+10)"
                    + " should not be accepted");
        } catch (OutOfRangeException oore) {
            // expected
        }
        //Check if null estimation technique can be injected
        try {
            Assert.assertNull(new Percentile(
                    (EstimationTechnique)null).getEstimationTechnique());
            Assert.fail("Unexpected: Percentile cannot have a NULL " +
                    "Estimation technique");
        }catch(NullArgumentException nae) {
            //expected as we cannot afford to have a null for estimation
        }
    }

    @Test
    public void testAllEstimationTechniquesOnly() {
        Assert.assertEquals("DEFAULT",DEFAULT.getName());
        Assert.assertEquals("Apache Commons",DEFAULT.getDescription());
        Object[][] map =
                new Object[][] { { DEFAULT, 20.82 }, { R1, 19.8 },
                        { R2, 19.8 }, { R3, 19.8 }, { R4, 19.310 },
                        { R7, 19.555 }, { R8, 20.460 } };
        for (Object[] o : map) {
            EstimationTechnique e = (EstimationTechnique) o[0];
            double expected = (Double) o[1];
            double result =
                    e.evaluate(testArray, testArray.length,
                            DEFAULT_PERCENTILE );
            Assert.assertEquals("expected[" + e + "] = " + expected +
                    " but was = " + result, expected, result, tolerance);
        }
    }

    @Test
    public void testAllEstimationTechniquesOnlyForExtremeIndexes() {
        final double MAX=100;
        Object[][] map =
                new Object[][] { { DEFAULT, 0d, MAX}, { R1, 0d,MAX+0.5 },
                { R2, 0d,MAX}, { R3, 0d,MAX }, { R4, 0d,MAX },
                { R7, 0d,MAX }, { R8, 0d,MAX }  };
        for (Object[] o : map) {
            EstimationTechnique e = (EstimationTechnique) o[0];
                Assert.assertEquals(((Double)o[1]).doubleValue(),
                        e.index(0d, (int)MAX),0d);
                Assert.assertEquals("Enum:"+e,((Double)o[2]).doubleValue(),
                        e.index(1.0, (int)MAX),0d);
            }
    }
    @Test
    public void testAllEstimationTechniquesOnlyForNullsAndOOR() {

        Object[][] map =
                new Object[][] { { DEFAULT, 20.82 }, { R1, 19.8 },
                        { R2, 19.8 }, { R3, 19.8 }, { R4, 19.310 },
                        { R7, 19.555 }, { R8, 20.460 } };
        for (Object[] o : map) {
            EstimationTechnique e = (EstimationTechnique) o[0];
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
     * @param map of expected result against a {@link EstimationTechnique}
     * @param p the quantile to compute for
     * @param tolerance the tolerance of difference allowed
     */
    protected void testAssertMappedValues(double[] d, Object[][] map, Double p,
            Double tolerance) {
        for (Object[] o : map) {
            EstimationTechnique e = (EstimationTechnique) o[0];
            double expected = (Double) o[1];
            double result = getUnivariateStatistic(p, e).evaluate(d);
            Assert.assertEquals("expected[" + e + "] = " + expected +
                    " but was = " + result, expected, result, tolerance);
        }
    }

}
