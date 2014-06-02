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

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.ListIterator;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.commons.math3.exception.MathIllegalArgumentException;
import org.apache.commons.math3.exception.NotFiniteNumberException;
import org.apache.commons.math3.exception.NullArgumentException;
import org.apache.commons.math3.exception.OutOfRangeException;
import org.apache.commons.math3.exception.util.LocalizedFormats;
import org.apache.commons.math3.stat.descriptive.AbstractUnivariateStatistic;
import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.util.MathUtils;

/**
 * Provides percentile computation.
 * <p>
 * There are several commonly used methods for estimating percentiles (a.k.a.
 * quantiles) based on sample data.  For large samples, the different methods
 * agree closely, but when sample sizes are small, different methods will give
 * significantly different results.  The algorithm implemented here works as follows:
 * <ol>
 * <li>Let <code>n</code> be the length of the (sorted) array and
 * <code>0 < p <= 100</code> be the desired percentile.</li>
 * <li>If <code> n = 1 </code> return the unique array element (regardless of
 * the value of <code>p</code>); otherwise </li>
 * <li>Compute the estimated percentile position
 * <code> pos = p * (n + 1) / 100</code> and the difference, <code>d</code>
 * between <code>pos</code> and <code>floor(pos)</code> (i.e. the fractional
 * part of <code>pos</code>).</li>
 * <li> If <code>pos < 1</code> return the smallest element in the array.</li>
 * <li> Else if <code>pos >= n</code> return the largest element in the array.</li>
 * <li> Else let <code>lower</code> be the element in position
 * <code>floor(pos)</code> in the array and let <code>upper</code> be the
 * next element in the array.  Return <code>lower + d * (upper - lower)</code>
 * </li>
 * </ol></p>
 * <p>
 * To compute percentiles, the data must be at least partially ordered.  Input
 * arrays are copied and recursively partitioned using an ordering definition.
 * The ordering used by <code>Arrays.sort(double[])</code> is the one determined
 * by {@link java.lang.Double#compareTo(Double)}.  This ordering makes
 * <code>Double.NaN</code> larger than any other value (including
 * <code>Double.POSITIVE_INFINITY</code>).  Therefore, for example, the median
 * (50th percentile) of
 * <code>{0, 1, 2, 3, 4, Double.NaN}</code> evaluates to <code>2.5.</code></p>
 * <p>
 * Since percentile estimation usually involves interpolation between array
 * elements, arrays containing  <code>NaN</code> or infinite values will often
 * result in <code>NaN</code> or infinite values returned.</p>
 * <p>
 * Since 2.2, Percentile uses only selection instead of complete sorting
 * and caches selection algorithm state between calls to the various
 * {@code evaluate} methods. This greatly improves efficiency, both for a single
 * percentile and multiple percentile computations. To maximize performance when
 * multiple percentiles are computed based on the same data, users should set the
 * data array once using either one of the {@link #evaluate(double[], double)} or
 * {@link #setData(double[])} methods and thereafter {@link #evaluate(double)}
 * with just the percentile provided.
 * </p>
 * <p>
 * <strong>Note that this implementation is not synchronized.</strong> If
 * multiple threads access an instance of this class concurrently, and at least
 * one of the threads invokes the <code>increment()</code> or
 * <code>clear()</code> method, it must be synchronized externally.</p>
 *
 * @version $Id: Percentile.java 1416643 2012-12-03 19:37:14Z tn $
 */
public class Percentile extends AbstractUnivariateStatistic implements Serializable {

    /** Serializable version identifier */
    private static final long serialVersionUID = -8091216485095130416L;

    /** Minimum size under which we use a simple insertion sort rather than Hoare's select. */
    private static final int MIN_SELECT_SIZE = 15;

    /** Maximum number of partitioning pivots cached (each level double the number of pivots). */
    private static final int MAX_CACHED_LEVELS = 10;

    /** Any of the {@link EstimationTechnique}s such as DEFAULT can be used. */
    private final EstimationTechnique estimationTechnique;

    /** Determines what percentile is computed when evaluate() is activated
     * with no quantile argument */
    private double quantile = 0.0;

    /** Cached pivots. */
    private int[] cachedPivots;

    /**
     * Constructs a Percentile with a default quantile
     * value of 50.0.
     */
    public Percentile() {
        // No try-catch or advertised exception here - arg is valid
        this(50.0);
    }

    /**
     * Constructs a Percentile with the specific quantile value.
     * @param p the quantile
     * @throws MathIllegalArgumentException  if p is not greater than 0 and less
     * than or equal to 100
     */
    public Percentile(final double p) throws MathIllegalArgumentException {
        this(p, EstimationTechnique.DEFAULT);//uses DEFAULT Estimation
    }

    /**
     * Copy constructor, creates a new {@code Percentile} identical
     * to the {@code original}
     *
     * @param original the {@code Percentile} instance to copy
     * @throws NullArgumentException if original is null
     */
    public Percentile(Percentile original) throws NullArgumentException {
        estimationTechnique = original.getEstimationTechnique();
        copy(original, this);
    }

    /**
     * Constructs a percentile with the specific quantile value and an
     * {@link EstimationTechnique}.
     *
     * @param p the quantile to be computed
     * @param technique one of the percentile {@link EstimationTechnique
     *            estimation techniques}
     * @throws MathIllegalArgumentException if p is not greater than 0 and less
     *             than or equal to 100
     * @throws NullArgumentException if technique passed in null
     */
    public Percentile(final double p, EstimationTechnique technique)
            throws MathIllegalArgumentException {
        setQuantile(p);
        cachedPivots = null;
        MathUtils.checkNotNull(technique);
        estimationTechnique = technique;
    }

    /**
     * Constructs a default percentile with a specific
     * {@link EstimationTechnique}.
     *
     * @param technique one of the percentile {@link EstimationTechnique
     *            estimation techniques}
     * @throws MathIllegalArgumentException if p is not greater than 0 and less
     *             than or equal to 100
     */
    public Percentile(EstimationTechnique technique)
            throws MathIllegalArgumentException {
        this(50, technique);
    }


    /** {@inheritDoc} */
    @Override
    public void setData(final double[] values) {
        if (values == null) {
            cachedPivots = null;
        } else {
            cachedPivots = new int[(0x1 << MAX_CACHED_LEVELS) - 1];
            Arrays.fill(cachedPivots, -1);
        }
        super.setData(values);
    }

    /** {@inheritDoc} */
    @Override
    public void setData(final double[] values, final int begin, final int length)
    throws MathIllegalArgumentException {
        if (values == null) {
            cachedPivots = null;
        } else {
            cachedPivots = new int[(0x1 << MAX_CACHED_LEVELS) - 1];
            Arrays.fill(cachedPivots, -1);
        }
        super.setData(values, begin, length);
    }

    /**
     * Returns the result of evaluating the statistic over the stored data.
     * <p>
     * The stored array is the one which was set by previous calls to
     * {@link #setData(double[])}
     * </p>
     * @param p the percentile value to compute
     * @return the value of the statistic applied to the stored data
     * @throws MathIllegalArgumentException if p is not a valid quantile value
     * (p must be greater than 0 and less than or equal to 100)
     */
    public double evaluate(final double p) throws MathIllegalArgumentException {
        return evaluate(getDataRef(), p);
    }

    /**
     * Returns an estimate of the <code>p</code>th percentile of the values
     * in the <code>values</code> array.
     * <p>
     * Calls to this method do not modify the internal <code>quantile</code>
     * state of this statistic.</p>
     * <p>
     * <ul>
     * <li>Returns <code>Double.NaN</code> if <code>values</code> has length
     * <code>0</code></li>
     * <li>Returns (for any value of <code>p</code>) <code>values[0]</code>
     *  if <code>values</code> has length <code>1</code></li>
     * <li>Throws <code>MathIllegalArgumentException</code> if <code>values</code>
     * is null or p is not a valid quantile value (p must be greater than 0
     * and less than or equal to 100) </li>
     * </ul></p>
     * <p>
     * See {@link Percentile} for a description of the percentile estimation
     * algorithm used.</p>
     *
     * @param values input array of values
     * @param p the percentile value to compute
     * @return the percentile value or Double.NaN if the array is empty
     * @throws MathIllegalArgumentException if <code>values</code> is null
     *     or p is invalid
     */
    public double evaluate(final double[] values, final double p)
    throws MathIllegalArgumentException {
        test(values, 0, 0);
        return evaluate(values, 0, values.length, p);
    }

    /**
     * Returns an estimate of the <code>quantile</code>th percentile of the
     * designated values in the <code>values</code> array.  The quantile
     * estimated is determined by the <code>quantile</code> property.
     * <p>
     * <ul>
     * <li>Returns <code>Double.NaN</code> if <code>length = 0</code></li>
     * <li>Returns (for any value of <code>quantile</code>)
     * <code>values[begin]</code> if <code>length = 1 </code></li>
     * <li>Throws <code>MathIllegalArgumentException</code> if <code>values</code>
     * is null, or <code>start</code> or <code>length</code> is invalid</li>
     * </ul></p>
     * <p>
     * See {@link Percentile} for a description of the percentile estimation
     * algorithm used.</p>
     *
     * @param values the input array
     * @param start index of the first array element to include
     * @param length the number of elements to include
     * @return the percentile value
     * @throws MathIllegalArgumentException if the parameters are not valid
     *
     */
    @Override
    public double evaluate(final double[] values, final int start, final int length)
    throws MathIllegalArgumentException {
        return evaluate(values, start, length, quantile);
    }

     /**
     * Returns an estimate of the <code>p</code>th percentile of the values
     * in the <code>values</code> array, starting with the element in (0-based)
     * position <code>begin</code> in the array and including <code>length</code>
     * values.
     * <p>
     * Calls to this method do not modify the internal <code>quantile</code>
     * state of this statistic.</p>
     * <p>
     * <ul>
     * <li>Returns <code>Double.NaN</code> if <code>length = 0</code></li>
     * <li>Returns (for any value of <code>p</code>) <code>values[begin]</code>
     *  if <code>length = 1 </code></li>
     * <li>Throws <code>MathIllegalArgumentException</code> if <code>values</code>
     *  is null , <code>begin</code> or <code>length</code> is invalid, or
     * <code>p</code> is not a valid quantile value (p must be greater than 0
     * and less than or equal to 100)</li>
     * </ul></p>
     * <p>
     * See {@link Percentile} for a description of the percentile estimation
     * algorithm used.</p>
     *
     * @param values array of input values
     * @param p  the percentile to compute
     * @param begin  the first (0-based) element to include in the computation
     * @param length  the number of array elements to include
     * @return  the percentile value
     * @throws MathIllegalArgumentException if the parameters are not valid or the
     * input array is null
     */
    public double evaluate(final double[] values, final int begin,
            final int length, final double p) throws MathIllegalArgumentException {

        test(values, begin, length);
        if (p > 100 || p <= 0) {
            throw new OutOfRangeException(
                    LocalizedFormats.OUT_OF_BOUNDS_QUANTILE_VALUE, p, 0, 100);
        }
        if (length == 0) {
            return Double.NaN;
        }
        if (length == 1) {
            return values[begin]; // always return single value for n = 1
        }
        final double[] work = getWorkArray(values, begin, length);
        final int[] pivotsHeap = getPivots(values);
        return estimationTechnique.evaluate(work, pivotsHeap, length, p);
    }


    /** Select a pivot index as the median of three
     * @param work data array
     * @param begin index of the first element of the slice
     * @param end index after the last element of the slice
     * @return the index of the median element chosen between the
     * first, the middle and the last element of the array slice
     * @deprecated Please refrain from using this method as this pivoting
                   strategy is modeled elsewhere
     */
    @Deprecated
    int medianOf3(final double[] work, final int begin, final int end) {
        return new KthSelector(work).pivotIndex(begin, end);
    }


    /**
     * Returns the value of the quantile field (determines what percentile is
     * computed when evaluate() is called with no quantile argument).
     *
     * @return quantile
     */
    public double getQuantile() {
        return quantile;
    }

    /**
     * Sets the value of the quantile field (determines what percentile is
     * computed when evaluate() is called with no quantile argument).
     *
     * @param p a value between 0 < p <= 100
     * @throws MathIllegalArgumentException  if p is not greater than 0 and less
     * than or equal to 100
     */
    public void setQuantile(final double p) throws MathIllegalArgumentException {
        if (p <= 0 || p > 100) {
            throw new OutOfRangeException(
                    LocalizedFormats.OUT_OF_BOUNDS_QUANTILE_VALUE, p, 0, 100);
        }
        quantile = p;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Percentile copy() {
        Percentile result = new Percentile();
        //No try-catch or advertised exception because args are guaranteed non-null
        copy(this, result);
        return result;
    }

    /**
     * Copies source to dest.
     * <p>Neither source nor dest can be null.</p>
     *
     * @param source Percentile to copy
     * @param dest Percentile to copy to
     * @throws NullArgumentException if either source or dest is null
     */
    public static void copy(Percentile source, Percentile dest)
        throws NullArgumentException {
        MathUtils.checkNotNull(source);
        MathUtils.checkNotNull(dest);
        dest.setData(source.getDataRef());
        if (source.cachedPivots != null) {
            System.arraycopy(source.cachedPivots, 0, dest.cachedPivots, 0, source.cachedPivots.length);
        }
        dest.quantile = source.quantile;
    }

    /**
     * Get the work array to operate
     *
     * @param values the array of numbers
     * @param begin index to start reading the array
     * @param length the length of array to be read from the begin index
     * @return work array sliced from values in the range [begin,begin+length)
     */
    private double[] getWorkArray(double[] values, final int begin,
            final int length) {
        double[] work;
        if (values == getDataRef()) {
            work = getDataRef();
        } else {
            work = new double[length];
            System.arraycopy(values, begin, work, 0, length);
        }
        return work;
    }

    /**
     * Get Pivots either cached or create one
     *
     * @param values array containing the input numbers
     * @return cached pivots or a newly created one
     */
    private int[] getPivots(final double[] values) {
        final int[] pivotsHeap;
        if (values == getDataRef()) {
            pivotsHeap = cachedPivots;
        } else {
            pivotsHeap = new int[(0x1 << MAX_CACHED_LEVELS) - 1];
            Arrays.fill(pivotsHeap, -1);
        }
        return pivotsHeap;
    }

    /**
     * Get the estimation technique set
     *
     * @return the estimationTechnique
     */
    public EstimationTechnique getEstimationTechnique() {
        return estimationTechnique;
    }

    /**
     * An enum on percentile estimation strategy as elucidated in <a
     * href=http://en.wikipedia.org/wiki/Quantile>wikipedia on quantile</a>. The
     * enum names are based on the techniques mentioned in wikipedia.
     * <p>
     * <b>Please note</b> whereever wikipedia mentioned in this enum's context;
     * it is actually referring to <a
     * href=http://en.wikipedia.org/wiki/Quantile>this page</a>.
     * <p>
     * Each enum has a MathJax comment about the formulaes used for index and
     * estimate that is directly reffered from wikipedia for ready reference. <br>
     * While this enum provides a pre-processing function in
     * {@link #evaluate(double[], int[], int, double) evaluate} method for the
     * input array, each specific enum may over-ride this to match computation
     * closely with R script output
     * <p>
     * Each of these technique specializes in 2 aspects viz index and estimate
     * <ll>
     * <li>An index to calculate rough approximate index of estimated percentile
     * <li>An estimate to interpolate/average/aggregate of percentile(s) found.
     * </ll>
     * <p>
     * Users can now create percentile with explicit setting of this enum.
     * <p>
     * Reference : <a
     * href=tolstoy.newcastle.edu.au/R/e17/help/att-1067/Quartiles_in_R
     * .pdf>Quartiles_in_R.pdf</a> has custom simplified R definitions of R1,
     * R2, R3 for quick reference
     */
    public static enum EstimationTechnique {
        /**
         * This is the default technique used in the {@link Percentile Apache
         * Commons Math Percentile}
         */
        DEFAULT("DEFAULT", "Apache Commons") {
            {
                {
                    exclusions=Collections.emptySet();
                }
            }

            @Override
            protected double index(final double p, final int N) {
                return Double.compare(p, 0d) == 0 ? 0 :
                       Double.compare(p, 1d) == 0 ? N : p * (N + 1);
            }

            /**
             * {@inheritDoc}. The DEFAULT technique does'nt need filtering and
             * hence dummied out
             */
            @Override
            protected double[] preProcess(final double[] work,
                    final AtomicInteger lengthHolder) {
                return work;
            }
        },
        /**
         * The method R1 is also referred as SAS-3 and has the following
         * formulaes for index and estimates<br>
         * index(<i>h</i>)=\( Np + 1/2\, \)<br>
         * estimate(<i>Q<sub>p</sub></i>)=\( x_{\lceil h\,-\,1/2 \rceil} \).
         * <p>
         * Simplified R perspective from references:
         *
         * <pre>
         * QuantileType1 <- function (v, p) {
         *                                      v = sort(v)
         *                                      m = 0
         *                                      n = length(v)
         *                                      j = floor((n * p) + m)
         *                                      g = (n * p) + m - j
         *                                      y = ifelse (g == 0, 0, 1)
         *                                      ((1 - y) * v[j]) + (y * v[j+1])
         *                                  }
         * </pre>
         */
        R1("R-1", "SAS-3") {

            @Override
            protected double index(final double p, final int N) {
                return Double.compare(p, 0d) == 0 ? 0 : N * p + 0.5;
            }

            /**
             * {@inheritDoc}This method is overridden to use ceil(pos-0.5)
             */
            @Override
            protected double estimate(final double[] values,
                    final int[] pivotsHeap, final double pos, final int length) {
                return super.estimate(values, pivotsHeap, Math.ceil(pos - 0.5),
                        length);
            }

        },
        /**
         * The method R2 is also referred as SAS-5 and has the following
         * formulaes for index and estimates<br>
         * index(<i>h</i>)=\( Np + 1/2\, \)<br>
         * estimate(<i>Q<sub>p</sub></i>)=\( \frac{x_{\lceil h\,-\,1/2 \rceil} +
         * x_{\lfloor h\,+\,1/2 \rfloor}}{2} \)
         * <p>
         * Simplified Type2 R Code perspective from references:
         *
         * <pre>
         * QuantileType2 <- function (v, p) {
         *                                     v = sort(v)
         *                                     m = 0
         *                                     n = length(v)
         *                                     j = floor((n * p) + m)
         *                                     g = (n * p) + m - j
         *                                     y = ifelse (g == 0, 0.5, 1)
         *                                     ((1 - y) * v[j]) + (y * v[j+1])
         *                                   }
         * </pre>
         */
        R2("R-2", "SAS-5") {

            @Override
            protected double index(final double p, final int N) {
                return Double.compare(p, 1d) == 0 ? N :
                       Double.compare(p, 0d) == 0 ? 0 : N * p + 0.5;
            }

            /**
             * {@inheritDoc}This technique in particular uses an average of
             * value at ceil(p+0.5) and floor(p-0.5).
             */
            @Override
            protected double estimate(final double[] values,
                    final int[] pivotsHeap, final double pos, final int length) {
                final double low =
                        super.estimate(values, pivotsHeap,
                                Math.ceil(pos - 0.5), length);
                final double high =
                        super.estimate(values, pivotsHeap,
                                Math.floor(pos + 0.5), length);
                return (low + high) / 2;
            }

        },
        /**
         * The method R3 is also referred as SAS-2 and has the following
         * formulaes for index and estimates<br>
         * index(<i>h</i>)=\( Np\, \)<br>
         * estimate(<i>Q<sub>p</sub></i>)=\( x_{\lfloor h \rceil}\, \)
         * <p>
         * Simplified Type 3 R code representation from references
         *
         * <pre>
         * QuantileType3 <- function (v, p) {
         *                                     v = sort(v)
         *                                     m = -0.5
         *                                     n = length(v)
         *                                     j = floor((n * p) + m)
         *                                     g = (n * p) + m - j
         *                                     y = ifelse(trunc(j/2)*2==j,
         *                                              ifelse(g==0, 0, 1), 1)
         *                                     ((1 - y) * v[j]) + (y * v[j+1])
         *                                   }
         * </pre>
         */
        R3("R-3", "SAS-2") {
            @Override
            protected double index(final double p, final int N) {
                return Double.compare(p, 0.5 / N) <= 0 ? 0 : Math.rint(N * p);
            }

        },
        /**
         * The method R4 is also referred as SAS-1 or ScyPy-(0,1) and has the
         * following formulaes for index and estimates<br>
         * index(<i>h</i>)=\( Np + 1/2\, \)<br>
         * estimate(<i>Q<sub>p</sub></i>)=\( x_{\lfloor h \rfloor} + (h -
         * \lfloor h \rfloor) (x_{\lfloor h \rfloor + 1} - x_{\lfloor h
         * \rfloor}) \)
         */
        R4("R-4", "SAS-1") {
            @Override
            protected double index(final double p, final int N) {
                return Double.compare(p, 1d / N) < 0 ? 0 :
                       Double.compare(p, 1) == 0 ? N : N * p;
            }

        },

        /**
         * The method implements Microsoft Excel style computation and is also
         * referred ScyPy-(1,1) and has the following formulaes for index and
         * estimates<br>
         * index(<i>h</i>)=\( (N-1)p + 1\, \)<br>
         * estimate(<i>Q<sub>p</sub></i>)=\( x_{\lfloor h \rfloor} + (h -
         * \lfloor h \rfloor) (x_{\lfloor h \rfloor + 1} - x_{\lfloor h
         * \rfloor}) \)
         * <p>
         * Simplified Type 7 R representation from references
         *
         * <pre>
         * QuantileType7 <- function (v, p) {
         *                                      v = sort(v)
         *                                      h = ((length(v)-1)*p)+1
         *                                      v[floor(h)]+  (
         *                                         (h-floor(h))*
         *                                         (v[floor(h)+1]- v[floor(h)])
         *                                      )
         *                                   }
         * </pre>
         */
        R7("R-7", "Excel") {
            @Override
            protected double index(final double p, final int N) {
                return Double.compare(p, 0d) == 0 ? 0 :
                       Double.compare(p, 1d) == 0 ? N : 1 + (N - 1) * p;
            }

        },

        /**
         * Most recommended approach as per wikipedia as it provides an unbiased
         * estimate which is distribution free.
         * <p>
         * The method is named R8 and is also referred ScyPy-(1/3,1/3) and has
         * the following formulaes for index and estimates<br>
         * index(<i>h</i>)=\( (N + 1/3)p + 1/3\, \)<br>
         * estimate(<i>Q<sub>p</sub></i>)=\( x_{\lfloor h \rfloor} + (h -
         * \lfloor h \rfloor) (x_{\lfloor h \rfloor + 1} - x_{\lfloor h
         * \rfloor}) \)
         */
        R8("R-8", "SciPy-(1/3,1/3)") {
            @Override
            protected double index(final double p, final int N) {
                double oneThird = 1d / 3;
                double nPlus = N + oneThird;
                double nMinus = N - oneThird;
                return Double.compare(p, 2 * oneThird / nPlus) < 0 ? 0 : Double
                        .compare(p, nMinus / nPlus) >= 0 ? N : nPlus * p +
                        oneThird;
            }

        };

        /** Exclusion Set which can be overriden*/
        protected Set<Double> exclusions=Collections.singleton(Double.NaN);

        /**
         * Simple name of the computation technique such as R-1, R-2. Please
         * refer to wikipedia to get the context of these names.
         */
        private String name;
        /**
         * An another description or alternate name for the computation
         * technique such as SAS-1,SAS-2. Please refer to wikipedia to get the
         * context of these names.
         */
        private String description;

        /**
         * Constructor
         *
         * @param type type of tecnique as per wikipedia
         * @param alternateName an another name for the percentile estimation
         *            technique that is mentioned in wikiepedia
         */
        EstimationTechnique(String type, String alternateName) {
            this.name = type;
            this.description = alternateName;
        }

        /**
         * Finds the index of array that can be used as starting value to find
         * and do interpolate value in
         * {@link #estimate(double[], int[], double, int) estimate} method.
         * <p>
         * This index calculation is specific to each computation technique and
         * hence is implemented specifically in each of EstimationTechnique
         * (enumerated as R1 - R9).
         *
         * @param p the pth quantile
         * @param N the total number of array elements in the work array
         * @return a computed real valued index as explained wikipedia
         */
        protected abstract double index(final double p, final int N);

        /**
         * Estimation based on kth selection. This may be overridden in
         * sub-classes/enums to compute slightly different estimations, such as
         * averaging etc.
         *
         * @param work array of numbers to be used for finding the percentile
         * @param pos indicated positional index prior computed from calling
         *            {@link #index(double, int)}
         * @param pivotsHeap a pre-populated cache if exists; will be used
         * @param length size of array considered
         * @return estimated percentile
         * @see #R2
         */
        protected double estimate(final double[] work, final int[] pivotsHeap,
                final double pos, final int length) {

            double fpos = FastMath.floor(pos);
            int intPos = (int) fpos;

            double dif = pos - fpos;
            KthSelector kthSelector = new KthSelector(work, pivotsHeap);
            if (pos < 1) {
                return kthSelector.select(0);
            }
            if (pos >= length) {
                return kthSelector.select(length - 1);
            }
            double lower = kthSelector.select(intPos - 1);
            double upper = kthSelector.select(intPos);
            return lower + dif * (upper - lower);
        }

        /**
         * Evaluate method to compute the percentile for a given bounded array.
         * This basically calls the {@link #index(double, int) index function}
         * and then calls {@link #estimate(double[], int[], double, int)
         * estimate function} to return the estimated percentile value.
         *
         * @param work array of numbers to be used for finding the percentile
         * @param pivotsHeap a prior cached heap which can speed up estimation
         * @param length the number of array elements to include
         * @param p the pth quantile to be computed
         * @return estimated percentile
         * @throws OutOfRangeException if length or p is out of range
         * @throws NullArgumentException if work array is null
         */
        protected double evaluate(final double[] work, final int[] pivotsHeap,
                final int length, final double p) {
            MathUtils.checkNotNull(work);
            if (p > 100 || p <= 0) {
                throw new OutOfRangeException(
                        LocalizedFormats.OUT_OF_BOUNDS_QUANTILE_VALUE,
                        p, 0, 100);
            }
            if (length < 0 || length > work.length) {
                throw new OutOfRangeException(length, 0, work.length);
            }
            final double quantile=p/100;
            final AtomicInteger lengthHolder=new AtomicInteger(length);
            final double[] newWork=preProcess(work,lengthHolder);
            return estimate(newWork, pivotsHeap,
                    index(quantile, lengthHolder.get()), lengthHolder.get());
        }

        /**
         * Evaluate method to compute the percentile for a given bounded array.
         * This basically calls the {@link #index(double, int) index function}
         * and then calls {@link #estimate(double[], int[], double, int)
         * estimate function} to return the estimated percentile value. Please
         * note that this method doesnt make use of precached pivots.
         *
         * @param work array of numbers to be used for finding the percentile
         * @param length the number of array elements to include
         * @param p the pth quantile to be computed
         * @return estimated percentile
         * @throws OutOfRangeException if length or p is out of range
         * @throws NullArgumentException if work array is null
         */
        public double evaluate(final double[] work, final int length,
                final double p) {
            return this.evaluate(work, null, length, p);
        }

        /**
         * A pre-process function to filter out un-needed elements. The sub
         * classes can do further filtering such as infinities etc if required. <br>
         * TODO: To abstract this out later if each enum has different need
         *
         * @param work the array containing the input numbers
         * @param lengthHolder a holder of length of work array which gets
         *            updated.
         * @return pre processed array and the length parameter updated
         * @throws OutOfRangeException if lengthHolder is out of range
         * @throws NullArgumentException if work array or lengthHolder is null
         */
        protected double[] preProcess(final double[] work,
                final AtomicInteger lengthHolder) {
            MathUtils.checkNotNull(work);
            MathUtils.checkNotNull(lengthHolder);
            final int length = lengthHolder.get();
            if (length < 0 || length > work.length) {
                throw new OutOfRangeException(length, 0, work.length);
            }
            double newWork[] = work;
            try {
                MathUtils.checkFinite(work);
            } catch (NotFiniteNumberException nfe) {
                // Filter out
                List<Double> l = new ArrayList<Double>();

                for (int i = 0; i < work.length; i++) {
                    l.add(work[i]);
                }
                for (ListIterator<Double> li = l.listIterator(); li.hasNext();) {
                    if (exclusions.contains(li.next())) {
                        li.remove();
                        lengthHolder.decrementAndGet();
                    }
                }
                newWork = new double[l.size()];
                for (int i = 0; i < l.size(); i++) {
                    newWork[i] = l.get(i);
                }
            }
            return newWork;
        }

        /**
         * Gets the name of the enum
         *
         * @return the name
         */
        String getName() {
            return name;
        }

        /**
         * Gets the description of the enum
         *
         * @return the description
         */
        String getDescription() {
            return description;
        }
    }

    /**
     * A Simple kth selector implementation to pick up the kth ordered element
     * from a work array containing the input numbers. This is used in the
     * context of computing percentile
     */
    private static class KthSelector {

        /**
         * A work array to use to find out the kth value
         */
        private final double[] work;

        /**
         * A pre-cached pivots that can be used for efficient estimation.
         */
        private final int[] pivotsHeap;

        /**
         * A {@link PivotingStrategy} used for pivoting
         */
        private final PivotingStrategy pivoting;

        /**
         * Constructor with no cached pivots
         *
         * @param values array containing input numbers
         */
        private KthSelector(final double[] values) {
            this(values, null);
        }

        /**
         * Constructor with cached pivots
         *
         * @param values array containing input numbers
         * @param pivots pivots that are pre-cached used for efficiency
         */
        private KthSelector(final double[] values, final int[] pivots) {
            this(values, pivots, PivotingStrategy.MEDIAN_OF_3);
        }

        /**
         * Constructor with soecified pivots cache and strategy
         *
         * @param values array containing imput numbers
         * @param pivots pivots that are pre-cached used for efficiency
         * @param pivotingStrategy one of the PivotingStrategys
         * @throws NullArgumentException
         */
        private KthSelector(final double[] values, final int[] pivots,
                PivotingStrategy pivotingStrategy) {
            MathUtils.checkNotNull(values);
            // MathUtils.checkNotNull(pivots);
            work = values;
            pivotsHeap = pivots;
            pivoting = pivotingStrategy;
        }

        /**
         * Select kth value in the array.
         *
         * @param k the index whose value in the array is of interest
         * @return kth value
         */
        protected double select(final int k) {
            int begin = 0;
            int end = work.length;
            int node = 0;
            boolean usePivotsHeap = pivotsHeap != null;
            while (end - begin > MIN_SELECT_SIZE) {
                final int pivot;

                if (usePivotsHeap && node < pivotsHeap.length &&
                        pivotsHeap[node] >= 0) {
                    // the pivot has already been found in a previous call
                    // and the array has already been partitioned around it
                    pivot = pivotsHeap[node];
                } else {
                    // select a pivot and partition work array around it
                    pivot =
                            partition(begin, end,
                                    pivoting.pivotIndex(work, begin, end));
                    if (usePivotsHeap && node < pivotsHeap.length) {
                        pivotsHeap[node] = pivot;
                    }
                }

                if (k == pivot) {
                    // the pivot was exactly the element we wanted
                    return work[k];
                } else if (k < pivot) {
                    // the element is in the left partition
                    end = pivot;
                    node =
                            FastMath.min(2 * node + 1, usePivotsHeap ?
                                    pivotsHeap.length : end);
                } else {
                    // the element is in the right partition
                    begin = pivot + 1;
                    node =
                            FastMath.min(2 * node + 2, usePivotsHeap ?
                                    pivotsHeap.length : end);
                }
            }
            Arrays.sort(work, begin, end);
            return work[k];
        }

        /**
         * A conveneient wrapper for pivotIndex
         *
         * @param begin start index to include
         * @param end ending index to include
         * @return pivot
         */
        private int pivotIndex(int begin, int end) {
            return pivoting.pivotIndex(work, begin, end);
        }

        /**
         * Partition an array slice around a pivot
         * <p>
         * Partitioning exchanges array elements such that all elements smaller
         * than pivot are before it and all elements larger than pivot are after
         * it
         * </p>
         *
         * @param begin index of the first element of the slice of work array
         * @param end index after the last element of the slice of work array
         * @param pivot initial index of the pivot
         * @return index of the pivot after partition
         */
        private int partition(final int begin, final int end, final int pivot) {

            final double value = work[pivot];
            work[pivot] = work[begin];

            int i = begin + 1;
            int j = end - 1;
            while (i < j) {
                while (i < j && work[j] > value) {
                    --j;
                }
                while (i < j && work[i] < value) {
                    ++i;
                }

                if (i < j) {
                    final double tmp = work[i];
                    work[i++] = work[j];
                    work[j--] = tmp;
                }
            }

            if (i >= end || work[i] > value) {
                --i;
            }
            work[begin] = work[i];
            work[i] = value;
            return i;
        }
    }

    /**
     * A strategy to choose pivoting index of an array for partitioning and
     * sorting. This is used for Kth selection/quick sorting. The strategy
     * allows a choice of techniques such as Median of 3, Random Pivot,
     * Central/End pivot ec.
     */
    private static enum PivotingStrategy {
        /**
         * This is the classic median of 3 approach for pivoting.
         */
        MEDIAN_OF_3() {
            /**
             * {@inheritDoc}.This in specific makes use of median of 3.
             * @param work data array
             * @param begin index of the first element of the slice
             * @param end index after the last element of the slice
             * @return the index of the pivot element chosen between the
             * first, middle and the last element of the array slice
             * @throws OutOfRangeException when indices exceeds range
             */
            @Override
            public int pivotIndex(final double[] work, final int begin,
                    final int end) {
                MathUtils.checkNotNull(work);
                if (begin < 0 || begin >= work.length) {
                    throw new OutOfRangeException(begin, 0, work.length);
                }
                if (end < begin || end > work.length) {
                    throw new OutOfRangeException(end, begin, work.length);
                }
                final int inclusiveEnd = end - 1;
                final int middle = begin + (inclusiveEnd - begin) / 2;
                final double wBegin = work[begin];
                final double wMiddle = work[middle];
                final double wEnd = work[inclusiveEnd];

                if (wBegin < wMiddle) {
                    if (wMiddle < wEnd) {
                        return middle;
                    } else {
                        return wBegin < wEnd ? inclusiveEnd : begin;
                    }
                } else {
                    if (wBegin < wEnd) {
                        return begin;
                    } else {
                        return wMiddle < wEnd ? inclusiveEnd : middle;
                    }
                }
            }
        },

        // RANDOM_PIVOT(){}
        // CENTRAL_PIVOT(){}
        ;

        /**
         * Find pivot index of the array so that partition and kth selection can
         * be made
         * @param work data array
         * @param begin index of the first element of the slice
         * @param end index after the last element of the slice
         * @return the index of the pivot element chosen between the
         * first and the last element of the array slice
         * @throws OutOfRangeException
         */
        protected abstract int pivotIndex(final double[] work, final int begin,
                final int end);
    }
}
