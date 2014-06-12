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

import org.apache.commons.math3.stat.descriptive.UnivariateStatistic;
import org.apache.commons.math3.stat.descriptive.UnivariateStatisticAbstractTest;
import org.apache.commons.math3.stat.descriptive.rank.Percentile.Type;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

/**
 * Test cases for the {@link UnivariateStatistic} class.
 * @version $Id: MedianTest.java 1244107 2012-02-14 16:17:55Z erans $
 */
public class MedianTest extends UnivariateStatisticAbstractTest{

    protected Median stat;

    /**
     * {@link  org.apache.commons.math3.stat.descriptive.rank.Percentile.Type type}
     *  to be used while calling
     * {@link #getUnivariateStatistic()}
     */
    protected Type estimationType = CM;

    /**
     * {@link Type}s that this test will verify against
     */
    protected final Type[] TYPES =
            new Type[] { CM, R_1, R_2, R_3, R_4, R_5, R_6, R_7, R_8,R_9 };


    /**
     * {@inheritDoc}
     */
    @Override
    public UnivariateStatistic getUnivariateStatistic() {
        return new Median();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double expectedValue() {
        return this.median;
    }

    @Before
    public void before() {
        estimationType=CM;
    }

    public UnivariateStatistic getUnivariateStatistic(Type e) {
        return new Median(e);
    }

    @Test
    public void testAllTechniquesSingleton() {
        double[] singletonArray = new double[] { 1d };
        for (Type e : TYPES) {
            UnivariateStatistic percentile = getUnivariateStatistic(e);
            Assert.assertEquals(1d, percentile.evaluate(singletonArray), 0);
            Assert.assertEquals(1d, percentile.evaluate(singletonArray, 0, 1),
                    0);
            Assert.assertEquals(1d,
                    new Median().evaluate(singletonArray, 0, 1, 5), 0);
            Assert.assertEquals(1d,
                    new Median().evaluate(singletonArray, 0, 1, 100), 0);
            Assert.assertTrue(Double.isNaN(percentile.evaluate(singletonArray,
                    0, 0)));
        }
    }
    @Test
    public void testAllTechniquesMedian() {
        double[] d = new double[] { 1, 3, 2, 4 };
        testAssertMappedValues(d, new Object[][] { { CM, 2.5d },
            { R_1, 2d }, { R_2, 2.5d }, { R_3, 2d }, { R_4, 2d }, { R_5, 2.5 },
            { R_6, 2.5 },{ R_7, 2.5 },{ R_8, 2.5 }, { R_9 , 2.5 } },  1.0e-05);

    }


    /**
     * Simple test assertion utility method
     *
     * @param d input data
     * @param map of expected result against a {@link Type}
     * @param tolerance the tolerance of difference allowed
     */
    protected void testAssertMappedValues(double[] d, Object[][] map,
            Double tolerance) {
        for (Object[] o : map) {
            Type e = (Type) o[0];
            double expected = (Double) o[1];
            double result = this.getUnivariateStatistic(e).evaluate(d);
            Assert.assertEquals("expected[" + e + "] = " + expected +
                    " but was = " + result, expected, result, tolerance);
        }
    }
}
