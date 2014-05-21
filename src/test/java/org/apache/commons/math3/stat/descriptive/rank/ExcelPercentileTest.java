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

import org.apache.commons.math3.exception.MathIllegalArgumentException;
import org.apache.commons.math3.stat.descriptive.UnivariateStatistic;
import org.apache.commons.math3.util.MathUtils;
import org.junit.Assert;
import org.junit.Test;


/**
 * Test cases to verify MS Excel style percentile calculation. The excel style
 * percentile values varies slightly from values by  {@link Percentile}. The
 * overriden class {@link ExcelPercentile} from {@link Percentile} is
 * encapsulated within this class.
 * @version $Id: ExcelPercentileTest.java 1244107 2012-02-14 16:17:55Z erans $
 */
public class ExcelPercentileTest extends PercentileTest{

    private final double percentile95=19.554999999999;
    private final double percentile5=8.81;
    /**
     * {@inheritDoc}
     */
    @Override
    public UnivariateStatistic getUnivariateStatistic() {
        return new ExcelPercentile(95.0);
    }

    @Override
    @Test
    public void testHighPercentile(){
        double[] d = new double[]{1, 2, 3};
        ExcelPercentile p = new ExcelPercentile(75);
        Assert.assertEquals(2.5/*3.0*/, p.evaluate(d), 1.0e-5);
    }

    @Override
    @Test
    public void testLowPercentile() {
        double[] d = new double[] {0, 1};
        ExcelPercentile p = new ExcelPercentile(25);
        Assert.assertEquals(0.25/*0d*/, p.evaluate(d), Double.MIN_VALUE);
    }

    @Override
    @Test
    public void testPercentile() {
        double[] d = new double[] {1, 3, 2, 4};
        ExcelPercentile p = new ExcelPercentile(30);
        Assert.assertEquals(1.9/*5*/, p.evaluate(d), 1.0e-5);
        p.setQuantile(25);
        Assert.assertEquals(1.75/*25*/, p.evaluate(d), 1.0e-5);
        p.setQuantile(75);
        Assert.assertEquals(3.25/*3.75*/, p.evaluate(d), 1.0e-5);
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

    @Override
    @Test
    public void testNISTExample() {
        double[] d = new double[] {95.1772, 95.1567, 95.1937, 95.1959,
                95.1442, 95.0610,  95.1591, 95.1195, 95.1772, 95.0925, 95.1990, 95.1682
        };
        ExcelPercentile p = new ExcelPercentile(90);
        Assert.assertEquals(95.19568/*95.1981*/, p.evaluate(d), 1.0e-4);
        Assert.assertEquals(95.1990, p.evaluate(d,0,d.length, 100d), 0);
    }

    @Override
    @Test
    public void test5() {
        ExcelPercentile percentile = new ExcelPercentile(5);
        Assert.assertEquals(this.percentile5, percentile.evaluate(testArray), getTolerance());
    }

    @Override
    @Test
    public void testNullEmpty() {
        ExcelPercentile percentile = new ExcelPercentile(50);
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

    @Override
    @Test
    public void testSingleton() {
        ExcelPercentile percentile = new ExcelPercentile(50);
        double[] singletonArray = new double[] {1d};
        Assert.assertEquals(1d, percentile.evaluate(singletonArray), 0);
        Assert.assertEquals(1d, percentile.evaluate(singletonArray, 0, 1), 0);
        Assert.assertEquals(1d, percentile.evaluate(singletonArray, 0, 1, 5), 0);
        Assert.assertEquals(1d, percentile.evaluate(singletonArray, 0, 1, 100), 0);
        Assert.assertTrue(Double.isNaN(percentile.evaluate(singletonArray, 0, 0)));
    }

    @Override
    @Test
    public void testSpecialValues() {
        ExcelPercentile percentile = new ExcelPercentile(50);
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

    @Override
    @Test
    public void testSetQuantile() {
        ExcelPercentile percentile = new ExcelPercentile(10);
        percentile.setQuantile(100); // OK
        Assert.assertEquals(100, percentile.getQuantile(), 0);
        try {
            percentile.setQuantile(0);
            Assert.fail("Expecting MathIllegalArgumentException");
        } catch (MathIllegalArgumentException ex) {
            // expected
        }
        try {
            new ExcelPercentile(0);
            Assert.fail("Expecting MathIllegalArgumentException");
        } catch (MathIllegalArgumentException ex) {
            // expected
        }
    }

    @Override
    public double expectedValue() {
        // TODO Auto-generated method stub
        return this.percentile95;
    }
    /**
     * Excel style percentile calculation as per excel formula explained in
     * http://support.microsoft.com/default.aspx?scid=kb;en-us;Q103493.
     * This class basically extends the default {@link Percentile} implement
     * however overloads required methods to realize the values closer to excel
     */
    private static class ExcelPercentile extends Percentile {
        /**
         * Serial Version ID
         */
        private static final long serialVersionUID = 4455832965852735084L;

        public ExcelPercentile() {
            super();
        }

        public ExcelPercentile(double p) throws MathIllegalArgumentException {
            super(p);
        }

        /**
         * {@inheritDoc}.<p> However this overriden method produces a quantile value,
         * which closely matches with Microsoft Excel way of producing percentile
         */
        @Override protected double computeQuantilePosition(double quantile,
                double length) {
            MathUtils.checkNotNull(length);
            MathUtils.checkNotNull(quantile);
            return 1+quantile*(length-1)/100;
        }
        /**
         * {@inheritDoc}
         */
        @Override
        public ExcelPercentile copy() {
            ExcelPercentile result = new ExcelPercentile();
            //No try-catch or advertised exception because args are guaranteed non-null
            copy(this, result);
            return result;
        }
    }

}
