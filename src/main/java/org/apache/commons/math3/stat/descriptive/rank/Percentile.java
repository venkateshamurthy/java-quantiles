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
import static org.apache.commons.math3.stat.descriptive.rank.Percentile.PivotingStrategy.MEDIAN_OF_3;

import java.io.Serializable;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.commons.math3.exception.MathIllegalArgumentException;
import org.apache.commons.math3.exception.MathUnsupportedOperationException;
import org.apache.commons.math3.exception.NotFiniteNumberException;
import org.apache.commons.math3.exception.NullArgumentException;
import org.apache.commons.math3.exception.OutOfRangeException;
import org.apache.commons.math3.exception.util.LocalizedFormats;
import org.apache.commons.math3.random.RandomDataGenerator;
import org.apache.commons.math3.stat.descriptive.AbstractUnivariateStatistic;
import org.apache.commons.math3.util.DoubleArray;
import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.util.MathArrays;
import org.apache.commons.math3.util.MathUtils;
import org.apache.commons.math3.util.ResizableDoubleArray;

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

    /** Maximum number of partitioning pivots cached (each level double the number of pivots). */
    private static final int MAX_CACHED_LEVELS = 10;

    /** Maximum number of cached pivots in the pivots cached array */
    private static final int PIVOTS_HEAP_LENGTH = 0x1 << MAX_CACHED_LEVELS - 1;

    /** Default pivoting strategy used while doing K<sup>th</sup> selection */
    private static PivotingStrategy pivotingStrategy = MEDIAN_OF_3;

    /** Any of the {@link Type}s such as DEFAULT can be used. */
    private final Type estimationType;

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
        this(p, Type.CM);//uses DEFAULT Estimation
    }

    /**
     * Copy constructor, creates a new {@code Percentile} identical
     * to the {@code original}
     *
     * @param original the {@code Percentile} instance to copy
     * @throws NullArgumentException if original is null
     */
    public Percentile(Percentile original) throws NullArgumentException {
        estimationType = original.getEstimationType();
        copy(original, this);
    }

    /**
     * Constructs a percentile with the specific quantile value and an
     * {@link Type}.
     *
     * @param p the quantile to be computed
     * @param type one of the percentile {@link Type
     *            estimation types}
     * @throws MathIllegalArgumentException if p is not greater than 0 and less
     *             than or equal to 100
     * @throws NullArgumentException if type passed in null
     */
    public Percentile(final double p, Type type)
            throws MathIllegalArgumentException {
        setQuantile(p);
        cachedPivots = null;
        MathUtils.checkNotNull(type);
        estimationType = type;
    }

    /**
     * Constructs a default percentile with a specific
     * {@link Type}.
     *
     * @param type one of the percentile {@link Type
     *            estimation types}
     * @throws NullArgumentException if type is null
     */
    public Percentile(Type type)
            throws NullArgumentException {
        this(50, type);
    }


    /** {@inheritDoc} */
    @Override
    public void setData(final double[] values) {
        if (values == null) {
            cachedPivots = null;
        } else {
            cachedPivots = new int[PIVOTS_HEAP_LENGTH];
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
            cachedPivots = new int[PIVOTS_HEAP_LENGTH];
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
        return estimationType.evaluate(work, pivotsHeap, length, p);
    }


    /** Select a pivot index as the median of three
     * <p>
     * <b>Note:</b> With the effect of allowing a strategy for
     * {@link PivotingStrategy pivoting} to be set on {@link Percentile} class;
     * this method is rendered inconsequential and hence will be unsupported.
     * @param work data array
     * @param begin index of the first element of the slice
     * @param end index after the last element of the slice
     * @return the index of the median element chosen between the
     * first, the middle and the last element of the array slice
     * @deprecated Please refrain from using this method and instead use
     * {@link Percentile#setPivotingStrategy(PivotingStrategy)} if required.
     * @throws MathUnsupportedOperationException when this method is invoked.
     *
     */
    @Deprecated
    int medianOf3(final double[] work, final int begin, final int end) {
        throw new MathUnsupportedOperationException();
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
    private double[] getWorkArray(final double[] values, final int begin,
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
     * Get pivots which is either cached or a newly created one
     *
     * @param values array containing the input numbers
     * @return cached pivots or a newly created one
     */
    private int[] getPivots(final double[] values) {
        final int[] pivotsHeap;
        if (values == getDataRef()) {
            pivotsHeap = cachedPivots;
        } else {
            pivotsHeap = new int[PIVOTS_HEAP_LENGTH];
            Arrays.fill(pivotsHeap, -1);
        }
        return pivotsHeap;
    }

    /**
     * Get the estimation type set
     *
     * @return the estimation type
     */
    public Type getEstimationType() {
        return estimationType;
    }

    /**
     * Get the {@link PivotingStrategy pivoting strategy} used
     * @return the pivotingStrategy set
     */
    public static PivotingStrategy getPivotingStrategy() {
        return pivotingStrategy;
    }

    /**
     * Set a {@link PivotingStrategy pivoting} strategy which can one of
     * <ul>
     * <li>{@link PivotingStrategy#MEDIAN_OF_3 MEDIAN_OF_3}</li>
     * <li>{@link PivotingStrategy#CENTRAL CENTRAL}</li>
     * <li>{@link PivotingStrategy#RANDOM RANDOM}</li>
     * <ul>
     * @param pivotingStrategy One of {@link PivotingStrategy} to be used
     * @throws NullArgumentException when pivotingStrategy is null
     */
    public static void setPivotingStrategy(
            final PivotingStrategy pivotingStrategy) {
        MathUtils.checkNotNull(pivotingStrategy);
        Percentile.pivotingStrategy = pivotingStrategy;
    }

    /**
     * An enum for various estimation strategies of a percentile referred in
     * <a href="http://en.wikipedia.org/wiki/Quantile">wikipedia on quantile</a>
     * with the names of enum matching those of types mentioned in
     * wikipedia.
     * <p>
     * Each enum corresponding to the specific type of estimation in wikipedia
     * implements  the respective formulae that specializes in the below aspects
     * <ul>
     * <li>An <b>index method</b> to calculate approximate index of the
     * estimate</li>
     * <li>An <b>estimate method</b> to estimate a value found at the earlier
     * computed index</li>
     * <li>A <b> minLimit</b> on the quantile for which first element of sorted
     * input is returned as an estimate </li>
     * <li>A <b> maxLimit</b> on the quantile for which last element of sorted
     * input is returned as an estimate </li>
     * </ul>
     * <p>
     * Users can now create {@link Percentile} by explicitly passing this enum;
     * such as by invoking {@link Percentile#Percentile(Type)}
     * <p>
     * References:
     * <ol>
     * <li>
     * <a href="http://en.wikipedia.org/wiki/Quantile">Wikipedia on quantile</a>
     * </li>
     * <li>
     * <a href="https://www.amherst.edu/media/view/129116/.../Sample+Quantiles.pdf">
     * Hyndman, R. J. and Fan, Y. (1996) Sample quantiles in statistical
     * packages, American Statistician 50, 361–365</a> </li>
     * <li>
     * <a href="http://stat.ethz.ch/R-manual/R-devel/library/stats/html/quantile.html">
     * R-Manual </a></li>
     * </ol>
     *
     */
    public static enum Type {
        /**
         * This is the default type used in the {@link Percentile}.This method
         * has the following formulae for index and estimates<br>
         * \( \begin{align}
         * &amp;index    = (N+1)p\ \\
         * &amp;estimate = x_{\lceil h\,-\,1/2 \rceil} \\
         * &amp;minLimit = 0 \\
         * &amp;maxLimit = 1 \\
         * \end{align}\)
         */
        CM("Commons Math") {
            {
                {
                    exclusions.clear();
                }
            }
            /**
             * {@inheritDoc}.This method in particular makes use of existing
             * Apache Commons Math style of picking up the index.
             */
            @Override
            protected double index(final double p, final int length) {
                final double minLimit = 0d;
                final double maxLimit = 1d;
                return Double.compare(p, minLimit) == 0 ? 0 :
                       Double.compare(p, maxLimit) == 0 ?
                               length : p * (length + 1);
            }

            /**
             * {@inheritDoc}. This type does'nt need filtering and
             * hence dummied out
             */
            @Override
            protected double[] preProcess(final double[] work,
                    final AtomicInteger lengthHolder) {
                return work;
            }
        },
        /**
         * The method R1 has the following formulae for index and estimates<br>
         * \( \begin{align}
         * &amp;index= Np + 1/2\,  \\
         * &amp;estimate= x_{\lceil h\,-\,1/2 \rceil} \\
         * &amp;minLimit = 0 \\
         * \end{align}\)
         */
        R_1("R-1") {

            @Override
            protected double index(final double p, final int length) {
                final double minLimit = 0d;
                return Double.compare(p, minLimit) == 0 ? 0 : length * p + 0.5;
            }

            /**
             * {@inheritDoc}This method in particular for R1 uses ceil(pos-0.5)
             */
            @Override
            protected double estimate(final double[] values,
                    final int[] pivotsHeap, final double pos,
                    final int length) {
                return super.estimate(values, pivotsHeap, Math.ceil(pos - 0.5),
                        length);
            }

        },
        /**
         * The method R2 has the following formulae for index and estimates<br>
         * \( \begin{align}
         * &amp;index= Np + 1/2\, \\
         * &amp;estimate=\frac{x_{\lceil h\,-\,1/2 \rceil} +
         * x_{\lfloor h\,+\,1/2 \rfloor}}{2} \\
         * &amp;minLimit = 0 \\
         * &amp;maxLimit = 1 \\
         * \end{align}\)
         */
        R_2("R-2") {

            @Override
            protected double index(final double p, final int length) {
                final double minLimit = 0d;
                final double maxLimit = 1d;
                return Double.compare(p, maxLimit) == 0 ? length :
                       Double.compare(p, minLimit) == 0 ? 0 : length * p + 0.5;
            }

            /**
             * {@inheritDoc}This method in particular for R2 averages the
             * values at ceil(p+0.5) and floor(p-0.5).
             */
            @Override
            protected double estimate(final double[] values,
                    final int[] pivotsHeap, final double pos,
                    final int length) {
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
         * The method R3 has the following formulae for index and estimates<br>
         * \( \begin{align}
         * &amp;index= Np \\
         * &amp;estimate= x_{\lfloor h \rceil}\, \\
         * &amp;minLimit = 0.5/N \\
         * \end{align}\)
         */
        R_3("R-3") {
            @Override
            protected double index(final double p, final int length) {
                final double minLimit = 1d/2 / length;
                return Double.compare(p, minLimit) <= 0 ?
                        0 : Math.rint(length * p);
            }

        },
        /**
         * The method R4 has the following formulae for index and estimates<br>
         * \( \begin{align}
         * &amp;index= Np\, \\
         * &amp;estimate= x_{\lfloor h \rfloor} + (h -
         * \lfloor h \rfloor) (x_{\lfloor h \rfloor + 1} - x_{\lfloor h
         * \rfloor}) \\
         * &amp;minLimit = 1/N \\
         * &amp;maxLimit = 1 \\
         * \end{align}\)
         */
        R_4("R-4") {
            @Override
            protected double index(final double p, final int length) {
                final double minLimit = 1d / length;
                final double maxLimit = 1d;
                return Double.compare(p, minLimit) < 0 ? 0 :
                       Double.compare(p, maxLimit) == 0 ? length : length * p;
            }

        },
        /**
         * The method R5 has the following formulae for index and estimates<br>
         * \( \begin{align}
         * &amp;index= Np + 1/2\\
         * &amp;estimate= x_{\lfloor h \rfloor} + (h -
         * \lfloor h \rfloor) (x_{\lfloor h \rfloor + 1} - x_{\lfloor h
         * \rfloor}) \\
         * &amp;minLimit = 0.5/N \\
         * &amp;maxLimit = (N-0.5)/N
         * \end{align}\)
         */
        R_5("R-5"){

            @Override
            protected double index(double p, int length) {
                final double minLimit = 1d/2 / length;
                final double maxLimit = (length - 0.5) / length;
                return Double.compare(p, minLimit) < 0 ? 0 :
                       Double.compare(p, maxLimit) >= 0 ?
                               length : length * p + 0.5;
            }
        },
        /**
         * The method R6 has the following formulae for index and estimates<br>
         * \( \begin{align}
         * &amp;index= (N + 1)p \\
         * &amp;estimate= x_{\lfloor h \rfloor} + (h -
         * \lfloor h \rfloor) (x_{\lfloor h \rfloor + 1} - x_{\lfloor h
         * \rfloor}) \\
         * &amp;minLimit = 1/(N+1) \\
         * &amp;maxLimit = N/(N+1) \\
         * \end{align}\)
         * <p>
         * <b>Note:</b> This method computes the index in a manner very close to
         * the default Commons Math Percentile existing implementation. However
         * the difference to be noted is in picking up the limits with which
         * first element (p&lt;1(N+1)) and last elements (p&gt;N/(N+1))are done.
         * While in default case; these are done with p=0 and p=1 respectively.
         */
        R_6("R-6"){

            @Override
            protected double index(double p, int length) {
                final double minLimit = 1d / (length + 1);
                final double maxLimit = 1d * length / (length + 1);
                return Double.compare(p, minLimit) < 0 ? 0 :
                       Double.compare(p, maxLimit) >= 0 ?
                               length : (length + 1) * p;
            }
        },

        /**
         * The method R7 implements Microsoft Excel style computation has the
         * following formulae for index and estimates.<br>
         * \( \begin{align}
         * &amp;index = (N-1)p + 1 \\
         * &amp;estimate = x_{\lfloor h \rfloor} + (h -
         * \lfloor h \rfloor) (x_{\lfloor h \rfloor + 1} - x_{\lfloor h
         * \rfloor}) \\
         * &amp;minLimit = 0 \\
         * &amp;maxLimit = 1 \\
         * \end{align}\)
         */
        R_7("R-7") {
            @Override
            protected double index(final double p, final int length) {
                final double minLimit = 0d;
                final double maxLimit = 1d;
                return Double.compare(p, minLimit) == 0 ? 0 :
                       Double.compare(p, maxLimit) == 0 ?
                               length : 1 + (length - 1) * p;
            }

        },

        /**
         * The method R8 has the following formulae for index and estimates<br>
         * \( \begin{align}
         * &amp;index = (N + 1/3)p + 1/3  \\
         * &amp;estimate = x_{\lfloor h \rfloor} + (h -
           \lfloor h \rfloor) (x_{\lfloor h \rfloor + 1} - x_{\lfloor h
         * \rfloor}) \\
         * &amp;minLimit = (2/3)/(N+1/3) \\
         * &amp;maxLimit = (N-1/3)/(N+1/3) \\
         * \end{align}\)
         * <p>
         * As per Ref [2,3] this approach is most recommended as it provides
         * an approximate median-unbiased estimate regardless of distribution.
         */
        R_8("R-8") {
            @Override
            protected double index(final double p, final int length) {
                final double minLimit = 2 * (1d / 3) / (length + 1d / 3);
                final double maxLimit =
                        (length - 1d / 3) / (length + 1d / 3);
                return Double.compare(p, minLimit) < 0 ? 0 :
                       Double.compare(p, maxLimit) >= 0 ? length :
                           (length + 1d / 3) * p + 1d / 3;
            }
        },

        /**
         * The method R9 has the following formulae for index and estimates<br>
         * \( \begin{align}
         * &amp;index = (N + 1/4)p + 3/8\\
         * &amp;estimate = x_{\lfloor h \rfloor} + (h -
           \lfloor h \rfloor) (x_{\lfloor h \rfloor + 1} - x_{\lfloor h
         * \rfloor}) \\
         * &amp;minLimit = (5/8)/(N+1/4) \\
         * &amp;maxLimit = (N-3/8)/(N+1/4) \\
         * \end{align}\)
         */
        R_9("R-9") {
            @Override
            protected double index(final double p, final int length) {
                final double minLimit = 5d/8 / (length + 0.25);
                final double maxLimit = (length - 3d/8) / (length + 0.25);
                return Double.compare(p, minLimit) < 0 ? 0 :
                       Double.compare(p, maxLimit) >= 0 ? length :
                               (length + 0.25) * p + 3d/8;
            }

        },
        ;

        /** Set of double values to be excluded from computation */
        @SuppressWarnings("serial")
        protected final Set<Double> exclusions = new HashSet<Double>()
                                                 {
                                                    {
                                                       add(Double.NaN);
                                                    }
                                                 };

        /** Simple name such as R-1, R-2 corresponding to those in wikipedia. */
        private final String name;

        /**
         * Constructor
         *
         * @param type name of estimation type as per wikipedia
         */
        Type(String type) {
            this.name = type;
        }

        /**
         * Finds the index of array that can be used as starting index to
         * {@link #estimate(double[], int[], double, int) estimate} percentile.
         * The index calculation is specific to each {@link Type}
         *
         * @param p the p<sup>th</sup> quantile
         * @param length the total number of array elements in the work array
         * @return a computed real valued index as explained in the wikipedia
         */
        protected abstract double index(final double p, final int length);

        /**
         * Estimation based on K<sup>th</sup> selection. This may be overridden
         * in specific enums to compute slightly different estimations.
         *
         * @param work array of numbers to be used for finding the percentile
         * @param pos indicated positional index prior computed from calling
         *            {@link #index(double, int)}
         * @param pivotsHeap an earlier populated cache if exists; will be used
         * @param length size of array considered
         * @return estimated percentile
         */
        protected double estimate(final double[] work, final int[] pivotsHeap,
                final double pos, final int length) {

            final double fpos = FastMath.floor(pos);
            final int intPos = (int) fpos;
            final double dif = pos - fpos;
            final KthSelector kthSelector = new KthSelector(work, pivotsHeap,
                    pivotingStrategy);

            if (pos < 1) {
                return kthSelector.select(0);
            }
            if (pos >= length) {
                return kthSelector.select(length - 1);
            }

            final double lower = kthSelector.select(intPos - 1);
            final double upper = kthSelector.select(intPos);
            return lower + dif * (upper - lower);
        }

        /**
         * Evaluate method to compute the percentile for a given bounded array
         * using earlier computed pivots heap.<br>
         * This basically calls the {@link #index(double, int) index function}
         * and then calls {@link #estimate(double[], int[], double, int)
         * estimate function} to return the estimated percentile value.
         *
         * @param work array of numbers to be used for finding the percentile
         * @param pivotsHeap a prior cached heap which can speed up estimation
         * @param length the number of array elements to include
         * @param p the p<sup>th</sup> quantile to be computed
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
         * note that this method does not make use of cached pivots.
         *
         * @param work array of numbers to be used for finding the percentile
         * @param length the number of array elements to include
         * @param p the p<sup>th</sup> quantile to be computed
         * @return estimated percentile
         * @throws OutOfRangeException if length or p is out of range
         * @throws NullArgumentException if work array is null
         */
        public double evaluate(final double[] work, final int length,
                final double p) {
            return this.evaluate(work, null, length, p);
        }

        /**
         * A function to filter out unneeded elements in the array. The sub
         * classes can do further filtering such as infinities if needed.
         *
         * @param work the array containing the input numbers
         * @param lengthHolder a holder of length of work array which gets
         *            updated and can be accessed by the caller.
         * @return processed array after required filtering
         * @throws MathIllegalArgumentException if lengthHolder is out of range
         * @throws NullArgumentException if work array or lengthHolder is null
         */
        protected double[] preProcess(final double[] work,
                final AtomicInteger lengthHolder) {

            MathUtils.checkNotNull(lengthHolder);
            final int length = lengthHolder.get();
            MathArrays.verifyValues(work,0,length);

            double[] newWork=work;
            try {
                MathUtils.checkFinite(work);
            } catch (NotFiniteNumberException nfe) {

                // Filter out only if exclusion is not empty
                if(!exclusions.isEmpty()) {
                    final DoubleArray array =
                            new ResizableDoubleArray(work.length);
                    for (double element:work) {
                        if(exclusions.contains(element)) {
                            lengthHolder.decrementAndGet();
                        }else {
                            array.addElement(element);
                        }
                    }
                    newWork=array.getElements();
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

    }

    /**
     * A Simple K<sup>th</sup> selector implementation to pick up the
     * K<sup>th</sup> ordered element from a work array containing the input
     * numbers.
     */
    private static class KthSelector {

        /** Minimum selection size for insertion sort rather than selection. */
        private static final int MIN_SELECT_SIZE = 15;

        /** A work array to use to find out the K<sup>th</sup> value */
        private final double[] work;

        /** A cached pivots heap that can be used for efficient estimation. */
        private final int[] pivotsHeap;

        /** A {@link PivotingStrategy} used for pivoting  */
        private final PivotingStrategy pivotingTechnique;

        /**
         * Constructor with specified pivots cache and pivoting strategy
         *
         * @param values array containing input numbers
         * @param pivots that are cached for efficient retrievals
         * @param pivotingStrategy any of the {@link PivotingStrategy}
         * @throws NullArgumentException when values or pivotingStrategy is null
         */
        private KthSelector(final double[] values, final int[] pivots,
                final PivotingStrategy pivotingStrategy) {
            MathUtils.checkNotNull(values);
            MathUtils.checkNotNull(pivotingStrategy);
            work = values;
            pivotsHeap = pivots;
            pivotingTechnique = pivotingStrategy;
        }

        /**
         * Select K<sup>th</sup> value in the array.
         *
         * @param k the index whose value in the array is of interest
         * @return K<sup>th</sup> value
         */
        protected double select(final int k) {
            int begin = 0;
            int end = work.length;
            int node = 0;
            final boolean usePivotsHeap = pivotsHeap != null;
            while (end - begin > MIN_SELECT_SIZE) {
                final int pivot;

                if (usePivotsHeap && node < pivotsHeap.length &&
                        pivotsHeap[node] >= 0) {
                    // the pivot has already been found in a previous call
                    // and the array has already been partitioned around it
                    pivot = pivotsHeap[node];
                } else {
                    // select a pivot and partition work array around it
                    pivot = partition(
                                begin,
                                end,
                                pivotingTechnique.pivotIndex(work, begin, end)
                            );
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
         * Partition an array slice around a pivot.Partitioning exchanges array
         * elements such that all elements smaller than pivot are before it and
         * all elements larger than pivot are after it.
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
     * A strategy to pick a pivoting index of an array for doing partitioning
     * and can be any of {@link PivotingStrategy#MEDIAN_OF_3},
     * {@link PivotingStrategy#RANDOM} or {@link PivotingStrategy#CENTRAL}.
     * This is used for K<sup>th</sup> element selection for the computation.
     */
    public static enum PivotingStrategy {

        /** Classic median of 3 strategy given begin and end indices. */
        MEDIAN_OF_3() {

            /**{@inheritDoc}
             * This in specific makes use of median of 3 pivoting.
             * @return The index corresponding to a pivot chosen between the
             * first, middle and the last indices of the array slice
             * @throws MathIllegalArgumentException when indices exceeds range
             */
            @Override
            public int pivotIndex(final double[] work, final int begin,
                    final int end) {
                MathArrays.verifyValues(work, begin, end-begin);
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

        /** A strategy of selecting random index between first and last index*/
        RANDOM(){

            /**
             * {@inheritDoc}
             * A uniform random pivot selection between begin and end indices
             * @return The index corresponding to a random uniformly selected
             * value between first and the last indices of the array slice
             * @throws MathIllegalArgumentException when indices exceeds range
             */
            @Override
            protected int pivotIndex(final double[] work, final int begin,
                    final int end) {
                MathArrays.verifyValues(work, begin, end-begin);
                return random.nextInt(begin, end-1);
            }
        },

        /** A mid point strategy based on the average of first and last index */
        CENTRAL(){

            /**
             * {@inheritDoc}
             * This in particular picks a average of begin and end indices
             * @return The index corresponding to a simple average of
             * the first and the last element indices of the array slice
             * @throws MathIllegalArgumentException when indices exceeds range
             */
            @Override
            protected int pivotIndex(final double[] work, final int begin,
                    final int end) {
                MathArrays.verifyValues(work, begin, end-begin);
                return begin + (end - begin)/2;
            }
        };

        /** A random data generator instance for randomized pivoting */
        protected final RandomDataGenerator random = new RandomDataGenerator();

        /**
         * Find pivot index of the array so that partition and K<sup>th</sup>
         * element selection can be made
         * @param work data array
         * @param begin index of the first element of the slice
         * @param end index after the last element of the slice
         * @return the index of the pivot element chosen between the
         * first and the last element of the array slice
         * @throws MathIllegalArgumentException when indices exceeds range
         */
        protected abstract int pivotIndex(final double[] work, final int begin,
                final int end);
    }
}
