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

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.Serializable;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.analysis.interpolation.LinearInterpolator;
import org.apache.commons.math3.analysis.interpolation.NevilleInterpolator;
import org.apache.commons.math3.analysis.interpolation.UnivariateInterpolator;
import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.exception.InsufficientDataException;
import org.apache.commons.math3.exception.NotANumberException;
import org.apache.commons.math3.exception.NullArgumentException;
import org.apache.commons.math3.exception.OutOfRangeException;
import org.apache.commons.math3.exception.util.LocalizedFormats;
import org.apache.commons.math3.stat.descriptive.AbstractStorelessUnivariateStatistic;
import org.apache.commons.math3.stat.descriptive.StorelessUnivariateStatistic;
import org.apache.commons.math3.util.MathArrays;
import org.apache.commons.math3.util.Precision;

//mvn site
/**
 * A percentile calculation technique implementing Apache
 * {@link StorelessUnivariateStatistic} which implements the <a
 * href=http://www.cs.wustl.edu/~jain/papers/ftp/psqr.pdf>P<SUP>2</SUP></a>
 * Algorithm as explained by <a href=http://www.cse.wustl.edu/~jain/>Raj
 * Jain</a> and Iimrich Chlamtac in their famous acclaimed work <a
 * href=http://www.cse.wustl.edu/~jain/papers/psqr.htm>P<SUP>2</SUP> Algorithm
 * for Dynamic Calculation of Quantiles and Histogram Without Storing
 * Observations</a>.
 * <p>
 * <b>Note: This implementation is not synchronized and produces an approximate
 * result. For lower number of samples (say <50 or 100) an exact Percentile can
 * be in place substituted </b>
 *
 * @version $Id$
 */
public class PSquaredPercentile extends AbstractStorelessUnivariateStatistic
implements StorelessUnivariateStatistic, Serializable {
    /**
     * The maximum array size used for psquare algorithm
     */
    static final int PSQUARE_CONSTANT = 5;

    /**
     * A Default quantile needed in case if user prefers to use default no
     * argument constructor.
     */
    private static final double DEFAULT_QUANTILE_DESIRED = 95d;
    /**
     * Serial ID
     */
    private static final long serialVersionUID = 2283912083175715479L;
    /**
     * An integer formatter for print convenience
     */
    private static final DecimalFormat INT_FORMAT = new DecimalFormat("00");

    /**
     * A decimal formatter for print convenience
     */
    private static final DecimalFormat DECIMAL_FORMAT = new DecimalFormat(
            "00.00");

    /**
     * Initial list of 5 numbers corresponding to 5 markers. <b>NOTE:</b>watch
     * out for the add methods that are overloaded
     */
    private List<Double> initialFive = new FixedCapacityList<Double>(
            PSQUARE_CONSTANT);

    /**
     * The quantile needed should be in range of 0-1. The constructor
     * {@link #PSquaredPercentile(double)} ensures that passed in percentile is
     * divide by 100
     */
    private final double quantile;

    /**
     * lastObservation is the last observation value/input sample. No need to
     * serialize
     */
    private transient double lastObservation;

    /**
     * The {@link PSquareEstimator} of quantile. No need to serialize
     */
    private transient PSquareEstimator estimator =
            new PSquareInterpolatorEvaluator();

    /**
     * {@link Markers} is the marker collection object which comes to effect
     * only after 5 values are inserted
     */
    private Markers markers = null;

    /**
     * Computed p value (i,e percentile value of data set hither to received)
     */
    private double pValue = Double.NaN;

    /**
     * Counter to count the values/observations accepted into this data set
     */
    private long countOfObservations;

    /**
     * Constructor with passed in required percentile that is within [0-100].
     *
     * @param percentile Desired percentile to be computed.Should be within
     *            [0-100] as this will be divided by 100 always.
     * @throws OutOfRangeException in case of percentile being asked is NOT
     *             within [0-100]
     */
    public PSquaredPercentile(final double percentile) {
        if (percentile > 100 || percentile < 0) {
            throw new OutOfRangeException(LocalizedFormats.OUT_OF_RANGE,
                    percentile, 0, 100);
        }
        this.quantile = percentile / 100d;// ALLWAYS GET IT WTIHIN [0,1]
    }

    /**
     * Default constructor that assumes a {@link #DEFAULT_QUANTILE_DESIRED
     * default quantile} needed
     */
    PSquaredPercentile() {
        this(DEFAULT_QUANTILE_DESIRED);
    }

    /**
     * {@inheritDoc} and any other attributes.
     */
    @Override
    public int hashCode() {
        double result = getResult();
        result = Double.isNaN(result) ? 37 : result;
        result = result + quantile;
        return (int) (result * 31 + getN() * 13);
    }

    /**
     * {@inheritDoc};However in addition in this class a check on the equality
     * of {@link Markers} is also made along with counts.
     */
    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o != null && o instanceof PSquaredPercentile) {
            PSquaredPercentile that = (PSquaredPercentile) o;
            boolean isNotNull = markers != null && that.markers != null;
            boolean isNull = markers == null  && that.markers == null;
            boolean result = isNotNull ? markers.equals(that.markers) : isNull;
            // markers as in the case of first
            // five observations
            return result && getN() == that.getN();
        } else {
            return false;
        }
    }

    /**
     * This method is used to set up some post construction initializations.
     *
     * @param aInputStream input stream to read from
     * @throws ClassNotFoundException thrown when class for deserializing is
     *             absent
     * @throws IOException thrown when an IO Error occurs during deserialization
     */
    private void readObject(ObjectInputStream aInputStream)
            throws ClassNotFoundException, IOException {
        aInputStream.defaultReadObject();
        estimator = new PSquareInterpolatorEvaluator(); // as this is transient
    }

    /**
     * Getter method for {@link markers}
     *
     * @return {@link #markers}
     */
    Markers markers() {
        return markers;
    }

    /**
     * {@inheritDoc}The internal state updated due to the new value in this
     * context is basically of the <a
     * href=http://www.cs.wustl.edu/~jain/papers/ftp/psqr.pdf>P<sup>2</sup></a>
     * algorithm marker positions and computation of the approximate quantile.
     * This method increments/accept a value / observation into the data set and
     * computes percentile. The result can be queried using {@link #getResult()}
     *
     * @param observation the observation currently being added.
     *
     */
    @Override
    public void increment(final double observation) {
        // Increment counter
        countOfObservations++;

        // Store last observation
        this.lastObservation = observation;

        // 0. Use Brute force for <5
        if (markers == null) {
            if (initialFive.add(observation)) {
                Collections.sort(initialFive);
                pValue = initialFive
                        .get((int) (quantile * (initialFive.size() - 1)));
                return;
            }
            // 1. Initialize once after 5th observation
            markers = new Markers(initialFive, quantile).estimator(estimator);
        }
        // 2. process a Data Point and return pValue
        pValue = markers.processDataPoint(observation);
    }

    /**
     * {@inheritDoc}. This populates a string containing the values of the
     * attributes all {@link Marker}s. Also adds a bit of formatting with
     * separators for convenience view.
     */
    @Override
    public String toString() {

        if (markers == null) {
            return String
                    .format("|%s |%s|%s|---------------|--------------------" +
                            "----------|-----------------------------|",
                            DECIMAL_FORMAT.format(lastObservation), "-",
                            DECIMAL_FORMAT.format(pValue));
        } else {
            return String.format("|%s %s",
                    DECIMAL_FORMAT.format(lastObservation), markers.toString());
        }
    }

    /**
     * {@inheritDoc}
     *
     * @see org.apache.commons.math3.stat.descriptive.StorelessUnivariateStatistic
     *      #getN()
     */
    public long getN() {
        return countOfObservations;
    }

    /**
     * {@inheritDoc}
     *
     * @see org.apache.commons.math3.stat.descriptive.AbstractStorelessUnivariateStatistic#copy()
     */
    @Override
    public StorelessUnivariateStatistic copy() {
        // multiply quantile by 100 now as anyway constructor divides it by 100
        double percentile = 100d * quantile;
        PSquaredPercentile copy = new PSquaredPercentile(percentile)
        .estimator(estimator);

        if (markers != null) {
            Markers marks = new Markers(initialFive, this.quantile())
            .initialize(markers.markerArray.clone()).estimator(
                    estimator);
            marks.postConstruct();
            copy = copy.markers(marks);
        }
        copy.countOfObservations = countOfObservations;
        copy.pValue = pValue;
        // TODO: I may need to replace this with a wrapper using
        // AbstractSerializableListDecorator (I cannot use FixedList here)
        copy.initialFive = new FixedCapacityList<Double>(PSQUARE_CONSTANT);
        copy.initialFive.addAll(initialFive);
        return copy;
    }

    /**
     * Sets the estimator
     *
     * @param theEstimator the {@link PSquareEstimator} to be set
     * @return this instance
     * @throws NullArgumentException
     */
    PSquaredPercentile estimator(PSquareEstimator theEstimator) {
        if (theEstimator == null) {
            throw new NullArgumentException(LocalizedFormats.NULL_NOT_ALLOWED);
        }
        this.estimator = theEstimator;
        return this;
    }

    /**
     * Sets the {@link Markers}
     *
     * @param theMarkers passed for setting it within
     * @return this
     */
    PSquaredPercentile markers(Markers theMarkers) {
        this.markers = theMarkers.estimator(estimator);
        return this;
    }

    /**
     * Returns the already set quantile desired which is in the range [0.0-1.0]
     *
     * @return quantile that is required to be computed for
     */
    public double quantile() {
        return quantile;
    }

    /**
     * {@inheritDoc}. This basically clears all the {@link #markers}, the
     * {@link #initialFive} list and sets {@link #countOfObservations} to 0
     *
     * @see org.apache.commons.math3.stat.descriptive.AbstractStorelessUnivariateStatistic#clear()
     */
    @Override
    public void clear() {
        markers = null;
        initialFive.clear();
        countOfObservations = 0L;
        pValue = Double.NaN;
    }

    /**
     * {@inheritDoc} This is basically the computed quantile value stored in
     * pValue.
     *
     * @see org.apache.commons.math3.stat.descriptive.AbstractStorelessUnivariateStatistic#getResult()
     */
    @Override
    public double getResult() {
        if (Double.compare(quantile, 1d) == 0) {
            pValue = maximum();
        } else if (Double.compare(quantile, 0d) == 0) {
            pValue = minimum();
        }
        return pValue;
    }

    /**
     * Return maximum in case of quantile=1
     *
     * @return maximum in the data set added to this statistic
     */
    private double maximum() {
        double val = Double.NaN;
        if (markers != null) {
            val = markers.markerArray[PSQUARE_CONSTANT].markerHeight;
        } else if (!initialFive.isEmpty()) {
            val = initialFive.get(initialFive.size() - 1);
        }
        return val;
    }

    /**
     * Return minimum in case of quantile=0
     *
     * @return minimum in the data set added to this statistic
     */
    private double minimum() {
        double val = Double.NaN;
        if (markers != null) {
            val = markers.markerArray[1].markerHeight;
        } else if (!initialFive.isEmpty()) {
            val = initialFive.get(0);
        }
        return val;
    }

    /**
     * Markers is an encapsulation of the five markers/buckets as indicated
     * in the original works of <a
     * href=http://www.cs.wustl.edu/~jain/papers/ftp/psqr.pdf>P<sup>2</sup></a>
     * algorithm.
     * <p>
     * This is a protected static class allowed for external enhancement and
     * testability.
     */
    protected static class Markers implements Serializable {
        /**
         * Serializable id
         */
        private static final long serialVersionUID = -8907278663345006480L;
        /**
         * Array of 5+1 {@link Marker}s (The first marker is dummy just so we
         * can match the rest of indexes [1-5] indicated in <a
         * href=http://www.cs.wustl.edu/~jain/papers/ftp/psqr.pdf>P<sup>2</sup>
         * algorithm paper</a> which follows unit based index)
         */
        private final Marker[] markerArray;
        /**
         * Kth cell belonging to [1-5] of the {@link #markerArray}. No need
         * for this to be serialized
         */
        private transient int k = -1;
        /**
         * The {@link PSquareEstimator} instance to be set every time. Not
         * serializing this
         */
        private transient PSquareEstimator estimator;

        /**
         * Constructor
         *
         * @param initial is a list of first five values
         * @param p is the quantile needed
         * @throws InsufficientDataException in case if initial list is null or
         *             having less than 5 elements
         *
         */
        Markers(List<Double> initial, double p) {
            int countObserved = initial == null ? -1 : initial.size();
            if (countObserved < PSQUARE_CONSTANT) {
                throw new InsufficientDataException(
                        LocalizedFormats.INSUFFICIENT_OBSERVED_POINTS_IN_SAMPLE,
                        countObserved, PSQUARE_CONSTANT);
            }
            Collections.sort(initial);
            initialize(markerArray = new Marker[] {
                    new Marker(),// Null Marker
                    new Marker(initial.get(0), 1, 0, 1),
                    new Marker(initial.get(1), 1 + 2 * p, p / 2, 2),
                    new Marker(initial.get(2), 1 + 4 * p, p, 3),
                    new Marker(initial.get(3), 3 + 2 * p, (1 + p) / 2, 4),
                    new Marker(initial.get(4), 5, 1, 5) });
        }

        /**
         * {@inheritDoc}
         */
        @Override
        public int hashCode() {
            int hash= estimator.hashCode() * 7;
            hash += markerArray[0].hashCode() * 31;
            hash += markerArray[1].hashCode() * 29;
            hash += markerArray[2].hashCode() * 23;
            hash += markerArray[3].hashCode() * 19;
            hash += markerArray[4].hashCode() * 17;
            hash += markerArray[5].hashCode() * 13;
            return hash;
        }

        /**
         * This equals method basically checks for {@link #markerArray} to be
         * deep equals and {@link #estimator} to be of same class.
         *
         * @param o is the other object
         * @return true if the object compares with this object are equivalent
         */
        @Override
        public boolean equals(Object o) {
            boolean result = false;
            if (this == o) {
                result = true;
            } else if (o != null && o instanceof Markers) {
                Markers that = (Markers) o;
                result = Arrays.deepEquals(markerArray, that.markerArray);
                result = result && estimator.getClass().isAssignableFrom(
                        that.estimator.getClass());
            }
            return result;
        }

        /**
         * Accessor for getting array of {@link Marker} that is
         * {@link #markerArray}
         *
         * @return {@link #markerArray}
         */
        Marker[] markerArray() {
            return markerArray;
        }

        /**
         * A Setter for setting the {@link PSquareEstimator} an instance of
         * {@link PSquareInterpolatorEvaluator}
         *
         * @param pSquareEstimator The {@link PSquareEstimator} to be set
         * @return this instance
         * @throws NullArgumentException when estimator passed is null
         */
        Markers estimator(PSquareEstimator pSquareEstimator) {
            if (pSquareEstimator == null) {
                throw new NullArgumentException(
                        LocalizedFormats.NULL_NOT_ALLOWED, "estimator");
            }
            this.estimator = pSquareEstimator;
            return this;
        }

        /**
         * Initialize markers with the passed {@link Marker} array
         *
         * @param markers passed in array of {@link Marker}
         * @return this instance
         */
        Markers initialize(Marker[] markers) {
            for (int i = 0; i < markers.length; i++) {
                markerArray[i].initialize(markers[i]);
            }
            postConstruct();
            return this;
        }

        /**
         * A post construct method which builds the links and initializes marker
         * indexes. It also sets the estimator.
         *
         */

        void postConstruct() {
            for (int i = 1; i < PSQUARE_CONSTANT; i++) {
                markerArray[i].previous(markerArray[i - 1])
                .next(markerArray[i + 1]).index(i);
            }
            markerArray[0].previous(markerArray[0]).next(markerArray[1])
            .index(0);
            markerArray[5].previous(markerArray[4]).next(markerArray[5])
            .index(5);
            if (estimator == null) {
                estimator = new PSquareInterpolatorEvaluator();
            }
        }

        /**
         * Process a data point
         *
         * @param inputDataPoint is the data point passed
         * @return computed percentile
         */
        public double processDataPoint(double inputDataPoint) {

            // 1. Find cell and update minima and maxima
            int kthCell = findCellAndUpdateMinMax(inputDataPoint);

            // 2. Increment positions
            incrementPositions(1, kthCell + 1, 5);

            // 2a. Update desired position with increments
            updateDesiredPositions();

            // 3. Adjust heights of m[2-4] if necessary
            adjustHeightsOfMarkers(estimator);

            // 4. Return percentile
            return getPercentileValue();
        }

        /**
         * @return {@link Marker#markerHeight} of mid point marker
         */
        public double getPercentileValue() {
            return markerArray[3].markerHeight;
        }

        /**
         * Finds the cell where the input observation / value fits
         *
         * @param observation the input value to be checked for
         * @return kth cell (of the markers ranging from 1-5) where observed
         *         sample fits
         */
        private int findCellAndUpdateMinMax(double observation) {
            k = -1;
            if (observation < markerArray[1].markerHeight) {
                markerArray[1].markerHeight = observation;
                k = 1;
            } else if (observation < markerArray[2].markerHeight) {
                k = 1;
            } else if (observation < markerArray[3].markerHeight) {
                k = 2;
            } else if (observation < markerArray[4].markerHeight) {
                k = 3;
            } else if (observation <= markerArray[5].markerHeight) {
                k = 4;
            } else {
                markerArray[5].markerHeight = observation;
                k = 4;
            }
            return k;
        }

        /**
         * Adjust marker heights by setting quantile estimates to middle markers
         *
         * @param pSquareEstimator The {@link PSquareEstimator} to be used for
         *            adjusting heights
         */
        private void adjustHeightsOfMarkers(PSquareEstimator pSquareEstimator) {
            for (int i = 2; i <= 4; i++) {
                markerArray[i].estimate(pSquareEstimator);
            }
        }

        /**
         * Increment positions by d. Please refer to algorithm paper for concept
         * of d and hence kept as d to represent it as delta
         *
         * @param d The increment value for the position
         * @param startIndex start index of the marker array
         * @param endIndex end index of the marker array
         */
        private void incrementPositions(int d, int startIndex, int endIndex) {
            for (int i = startIndex; i <= endIndex; i++) {
                markerArray[i].incrementPosition(d);
            }
        }

        /**
         * Desired positions incremented by bucket width. Bucket width is
         * basically the desired increments
         */
        private void updateDesiredPositions() {
            for (int i = 1; i < markerArray.length; i++) {
                markerArray[i].updateDesiredPosition();
            }
        }

        /**
         * This basically calls the {@link #postConstruct()} to set up the
         * linking indexes which is not captured while serializing.
         *
         * @param anInputStream the input stream to be deserialized
         * @throws ClassNotFoundException thrown when a desired class not found
         * @throws IOException thrown due to any io errors
         */
        private void readObject(ObjectInputStream anInputStream)
                throws ClassNotFoundException, IOException {
            // always perform the default de-serialization first
            anInputStream.defaultReadObject();
            postConstruct();
        }

        /**
         * toString
         *
         * @return string form
         */
        @Override
        public String toString() {

            return String.format(
                    "|%d|%s|%s|%s|%s|",
                    k,
                    DECIMAL_FORMAT.format(getPercentileValue()),
                    doubleToString(integerMarkerPositions(), INT_FORMAT, " "),
                    doubleToString(quantiles(), DECIMAL_FORMAT, " "),
                    doubleToString(desiredMarkerPositions(), DECIMAL_FORMAT,
                            " "));
        }

        /**
         * doubleToString provides a formatted representation of double array
         *
         * @param doubleArray the double data array
         * @param decimalFormat the formatter to be used for formatting
         * @param delimiter a delimiter between elements
         * @return text formatted representation of data array
         */
        private static String doubleToString(Double[] doubleArray,
                DecimalFormat decimalFormat, String delimiter) {
            StringBuilder sb = new StringBuilder();
            String formatterElement = "%s" + delimiter;
            Object[] values = new String[doubleArray.length];
            for (int i = 0; i < doubleArray.length; i++) {
                sb.append(formatterElement);
                values[i] = decimalFormat.format(doubleArray[i]);
            }
            return String.format(sb.toString(), values);
        }

        /**
         * The integral positions of all Markers as array
         *
         * @return an array of all of {@link Marker#integralMarkerPosition}s
         */
        public Double[] integerMarkerPositions() {

            return new Double[] {
                    Precision.round(markerArray[1].integralMarkerPosition, 0),
                    Precision.round(markerArray[2].integralMarkerPosition, 0),
                    Precision.round(markerArray[3].integralMarkerPosition, 0),
                    Precision.round(markerArray[4].integralMarkerPosition, 0),
                    Precision.round(markerArray[5].integralMarkerPosition, 0) };
        }

        /**
         * The desired positions of all markers as an array
         *
         * @return an array of all of {@link Marker#desiredMarkerPosition}s
         */
        public Double[] desiredMarkerPositions() {
            return new Double[] {
                    Precision.round(markerArray[1].desiredMarkerPosition, 2),
                    Precision.round(markerArray[2].desiredMarkerPosition, 2),
                    Precision.round(markerArray[3].desiredMarkerPosition, 2),
                    Precision.round(markerArray[4].desiredMarkerPosition, 2),
                    Precision.round(markerArray[5].desiredMarkerPosition, 2) };
        }

        /**
         * The quantile array of all the markers
         *
         * @return an array of all of {@link Marker#markerHeight}s
         */
        public Double[] quantiles() {
            return new Double[] {
                    Precision.round(markerArray[1].markerHeight, 2),
                    Precision.round(markerArray[2].markerHeight, 2),
                    Precision.round(markerArray[3].markerHeight, 2),
                    Precision.round(markerArray[4].markerHeight, 2),
                    Precision.round(markerArray[5].markerHeight, 2) };
        }

        /**
         * Returns the markers as a {@link Map}
         *
         * @return a map of marker attributes
         */
        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<String, Object>();
            for (Marker mark : markerArray) {
                map.putAll(mark.toMap());
            }
            return map;
        }

    }

    /**
     *
     * The class modeling the attributes of the marker of the <a
     * href=http://www.cs.wustl.edu/~jain/papers/ftp/psqr.pdf>P<sup>2</sup></a>
     * algorithm.
     * <p>
     * This is a protected static class for the reason it could be extended for
     * minor modification and also helps in testability
     */

    protected static class Marker implements Serializable {

        /**
         * Serial Version ID
         */
        private static final long serialVersionUID = -3575879478288538431L;

        /**
         * The marker index which is just a serial number for the marker in the
         * marker array of 5+1. Will be set during
         * {@link Markers#initialize(Marker[])}
         */
        private int index;

        /**
         * The integral marker position. Refer to the <a
         * href=http://www.cs.wustl.edu/~jain/papers/ftp/psqr.pdf>P<sup>2</sup>
         * algorithm paper</a> for the variable n
         */
        private double integralMarkerPosition;

        /**
         * Desired marker position. Refer to the <a
         * href=http://www.cs.wustl.edu/~jain/papers/ftp/psqr.pdf>P<sup>2</sup>
         * algorithm paper</a> for the variable n'
         */
        private double desiredMarkerPosition;

        /**
         * Marker height or the quantile. Refer to the <a
         * href=http://www.cs.wustl.edu/~jain/papers/ftp/psqr.pdf>P<sup>2</sup>
         * algorithm paper</a> for the variable q
         */
        private double markerHeight;

        /**
         * Desired marker increment. Refer to the <a
         * href=http://www.cs.wustl.edu/~jain/papers/ftp/psqr.pdf>P<sup>2</sup>
         * algorithm paper</a> for the variable dn'
         */
        private double desiredMarkerIncrement;

        /**
         * Next and previous markers for easy linked navigation in loops. this
         * is not serialized as they can be rebuilt during de-serialization.
         */
        private transient Marker next;
        /**
         * The previous marker links
         */
        private transient Marker previous;

        /**
         * Default constructor
         */
        public Marker() {
            this.next = this.previous = this;
        }

        /**
         * Constructor of a <a
         * href=http://www.cs.wustl.edu/~jain/papers/ftp/psqr.pdf>P<sup>2</sup>
         * algorithm</a> marker with following parameters
         *
         * @param heightOfMarker represent the quantile value
         * @param makerPositionDesired represent the desired marker position
         * @param markerPositionIncrement represent increments for position
         * @param markerPositionNumber represent the position number of marker
         */
        public Marker(double heightOfMarker, double makerPositionDesired,
                double markerPositionIncrement, double markerPositionNumber) {
            this.markerHeight = heightOfMarker;
            this.desiredMarkerPosition = makerPositionDesired;
            this.desiredMarkerIncrement = markerPositionIncrement;
            this.integralMarkerPosition = markerPositionNumber;
            this.next = this.previous = this; // initially self linked
        }

        /**
         * Navigates to the {@link #next} marker
         *
         * @return the next marker set
         */
        Marker next() {
            return next;
        }

        /**
         * Navigates to the {@link #previous} marker
         *
         * @return previous marker set
         */
        Marker previous() {
            return previous;
        }

        /**
         * Sets the {@link #previous} marker
         *
         * @param previousMarker the previous marker to the current marker in
         *            the array of markers
         * @return this instance
         */
        Marker previous(Marker previousMarker) {
            this.previous = previousMarker;
            return this;
        }

        /**
         * Sets the {@link #next} marker
         *
         * @param nextMarker the next marker to the current marker in the array
         *            of markers
         * @return this instance
         */
        Marker next(Marker nextMarker) {
            this.next = nextMarker;
            return this;
        }

        /**
         * Sets the {@link #index} of the marker
         *
         * @param indexOfMarker the array index of the marker in marker array
         * @return this instance
         */
        Marker index(int indexOfMarker) {
            this.index = indexOfMarker;
            return this;
        }

        /**
         * Initializes with passed in marker / copy construcor. No changes to
         * {@link #previous} and {@link #next} pointers.
         *
         * @param marker whose values will be used for initializing this
         *            instance
         * @return this instance of marker
         */
        Marker initialize(Marker marker) {
            return markerHeight(marker.markerHeight)
                    .desiredMarkerPosition(marker.desiredMarkerPosition)
                    .integralMarkerPosition(marker.integralMarkerPosition)
                    .desiredMarkerIncrement(marker.desiredMarkerIncrement);
        }

        /**
         * Sets quantile value to this instance which is also the height of
         * marker
         *
         * @param quantile the quantile to be set
         * @return this instance
         */
        Marker markerHeight(double quantile) {
            this.markerHeight = quantile;
            return this;
        }

        /**
         * Sets desired position ( a double quantity) of this marker
         *
         * @param markerPositionDesired the desired position (real number)
         * @return this instance
         */
        Marker desiredMarkerPosition(double markerPositionDesired) {
            this.desiredMarkerPosition = markerPositionDesired;
            return this;
        }

        /**
         * Sets integral position of this marker
         *
         * @param markerPositionNumber the integral numbered position
         * @return this instance
         */
        Marker integralMarkerPosition(double markerPositionNumber) {
            this.integralMarkerPosition = markerPositionNumber;
            return this;
        }

        /**
         * Sets the desired increment of a marker. Please refer to original
         * works to understand the concept of marker increment
         *
         * @param markerIncrementDesired the desired increment value of marker
         * @return this instance
         */
        Marker desiredMarkerIncrement(double markerIncrementDesired) {
            this.desiredMarkerIncrement = markerIncrementDesired;
            return this;
        }

        /**
         * Update desired Position with increment
         */
        void updateDesiredPosition() {
            desiredMarkerPosition += desiredMarkerIncrement;
        }

        /**
         * Increment Position by d
         *
         * @param d a delta value to increment
         */
        void incrementPosition(int d) {
            integralMarkerPosition += d;
        }

        /**
         * A name to the marker while representing in string form
         *
         * @return name
         */
        String name() {
            return "m" + index;
        }

        /**
         * Difference between desired and actual position
         *
         * @return difference between desired and actual position
         */
        double difference() {
            return desiredMarkerPosition - integralMarkerPosition;
        }

        /**
         * Builds a map of attribute names and their values and returns. the key
         * variable names n, np, q, dn are to be understood and inferred from
         * original works and the javadoc of {@link Marker}
         *
         * @return map containing keys (index, n ,np, q, dn with each of these
         *         prefixed with {@link #name()}) and their corresponding values
         *         contained in this instance
         */
        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<String, Object>();
            map.put(name() + ".index", (double) index);
            map.put(name() + ".n", Precision.round(integralMarkerPosition, 0));
            map.put(name() + ".np", Precision.round(desiredMarkerPosition, 2));
            map.put(name() + ".q", Precision.round(markerHeight, 2));
            map.put(name() + ".dn", Precision.round(desiredMarkerIncrement, 2));
            return map;
        }

        /**
         * {@inheritDoc}. This method basically returns a {@link #toMap() map
         * view} of the attributes in a text form
         */
        @Override
        public String toString() {
            return toMap().toString();
        }

        /**
         * estimate the quantile for the current marker
         *
         * @param estimator an instance of {@link PSquareEstimator} to be used
         *            for estimation
         * @return estimated quantile
         */
        public double estimate(PSquareEstimator estimator) {
            double di = difference();
            boolean isNextHigher =
                    next.integralMarkerPosition - integralMarkerPosition > 1;
            boolean isPreviousLower =
            previous.integralMarkerPosition - integralMarkerPosition < -1;

            if (di >= 1 && isNextHigher || di <= -1 && isPreviousLower){
               int d = di >= 0 ? 1 : -1;
               double[] x = new double[] { previous.integralMarkerPosition,
                          integralMarkerPosition, next.integralMarkerPosition };
               double[] y = new double[] { previous.markerHeight,
                          markerHeight, next.markerHeight };
               markerHeight =
                          estimator.estimate(x, y, integralMarkerPosition + d);
               incrementPosition(d);
            }
            return markerHeight;
        }

        /**
         * {@inheritDoc}<i>This equals method checks for equivalence of
         * {@link #integralMarkerPosition},{@link #desiredMarkerPosition},
         * {@link #desiredMarkerIncrement} and {@link #markerHeight} and as well
         * checks if navigation pointers({@link #next} and {@link #previous})
         * are the same between this and passed in object</i>
         *
         * @param o Other object
         * @return true if this equals passed in other object o
         */
        @Override
        public boolean equals(Object o) {
            boolean result = false;
            if (this == o) {
                result = true;
            } else if (o != null && o instanceof Marker) {
                Marker that = (Marker) o;

                boolean isSameHeight =
                        Double.compare(markerHeight, that.markerHeight) == 0;
                boolean isSamePosition =
                        Double.compare(integralMarkerPosition,
                                that.integralMarkerPosition) == 0;
                boolean isSameDesiredPosition =
                        Double.compare(desiredMarkerPosition,
                                that.desiredMarkerPosition) == 0;
                boolean isSameIncrement =
                        Double.compare(desiredMarkerIncrement,
                                that.desiredMarkerIncrement) == 0;

                result = isSameHeight  && isSamePosition;
                result = result && isSameDesiredPosition && isSameIncrement;

                boolean eitherNextNull = next == null || that.next == null;
                boolean eitherPrevNull =
                        previous == null || that.previous == null;
                boolean isNextEqual =
                        eitherNextNull ? true : next.index == that.next.index;
                boolean isPrevEqual =
                        eitherPrevNull ?
                                true : previous.index == that.previous.index;

                result = result  && isNextEqual && isPrevEqual;
            }
            return result;
        }

        @Override
        public int hashCode() {
            int hash = (int) (markerHeight + integralMarkerPosition);
            hash += desiredMarkerPosition + desiredMarkerIncrement;
            hash = hash * 31 + previous.index * 23 + next.index * 19;
            return hash;
        }

        /**
         * This method is used to set up some post construction initializations.
         *
         * @param anInputStream input stream to read from
         * @throws ClassNotFoundException thrown when class for deserialization
         *             is absent
         * @throws IOException thrown when an IO Error occurs.
         */
        private void readObject(ObjectInputStream anInputStream)
                throws ClassNotFoundException, IOException {
            anInputStream.defaultReadObject();
            previous = next = this;
        }

    }

    /**
     * An Interface modeling to estimate quantile based on <a
     * href=http://www.cs.wustl.edu/~jain/papers/ftp/psqr.pdf>P<sup>2</sup>
     * algorithm</a> which require 3 points to fit a parabolic curve. It also
     * makes use of polynomial interpolation to find an estimate of the
     * quantile;however switches to a linear interpolation in case if polynomial
     * interpolation results in an undesirable results.Please refer to details
     * in the <a
     * href=http://www.cs.wustl.edu/~jain/papers/ftp/psqr.pdf>P<sup>2</sup>
     * algorithm paper</a> as to when the switch happens to linear
     * interpolation.
     * <p>
     * The reason for making this as protected interface is to allow for future
     * possibilities of implementing it differently than the one available here
     * or composing with different polynomial interpolations.
     *
     */
    protected interface PSquareEstimator {
        /**
         * Estimates the quantile at a point based on passed in array of points
         * specified by
         * (x<sub>0</sub>,y<sub>0</sub>),(x<sub>1</sub>,y<sub>1</sub>),
         * (x<sub>2</sub>,y<sub>2</sub>) in a two dimensional plane.
         *
         * @param x a 3 element array corresponding to x coordinates of the
         *            three points (x[0], x[1], x[2])
         * @param y a 3 element array corresponding to y coordinates of the
         *            three points (y[0], y[1], y[2])
         * @param z a x value off the value of x[1] by a distance d
         * @return the p<sup>2</sup> estimation value of quantile at a value z
         */
        double estimate(double[] x, double[] y, double z);

        /**
         * Returns the number of times quadratic estimate was used finally
         *
         * @return the count of quadratic estimate used finally
         */
        int quadraticEstimationCount();

        /**
         * Returns the number of times linear estimate was chosen finally
         *
         * @return the count of linear estimate used finally
         */
        int linearEstimationCount();
    }

    /**
     * This is a modular implementation of <a
     * href=http://www.cs.wustl.edu/~jain/papers/ftp/psqr.pdf>P<SUP>2</SUP>
     * Algorithm</a> estimation by delegating to existing instances of
     * {@link UnivariateInterpolator} such as {@link NevilleInterpolator} for
     * the first estimation based on quadratic formulae and a
     * {@link LinearInterpolator} in case if needed for correction as is
     * reasoned out in the <a
     * href=http://www.cs.wustl.edu/~jain/papers/ftp/psqr.pdf>P<SUP>2</SUP>
     * Algorithm</a> for the choice of quadratic vs linear estimation
     * <p>
     * This is a protected static class allowing for possible enhancements on
     * the univariate interpolation and as well helps in testability
     *
     */
    protected static class PSquareInterpolatorEvaluator implements
    UnivariateInterpolator, PSquareEstimator, Serializable {
        /**
         * Serial ID
         */
        private static final long serialVersionUID = -5034313858440101891L;
        /**
         * A non linear interpolator such as {@link NevilleInterpolator} that
         * first provides an interpolated value which will be tested for bounds.
         */
        private transient UnivariateInterpolator nonLinear =
                new NevilleInterpolator();
        /**
         * A linear interpolator used to correct in case if non linear
         * interpolated gives an out of bounds.
         */
        private transient UnivariateInterpolator linear =
                new LinearInterpolator();
        /**
         * linearEstimationCount counts the number of times for linear
         * estimation used.
         */
        private transient int linearEstimationCount = 0;
        /**
         * quadraticEstimationCount counts the number of times for non linear
         * estimation used.(q actually means quadratic).
         */
        private transient int quadraticEstimationCount = 0;
        /**
         * xD is the desired x+dth position for which quantile needs to be
         * calculated. The increment d is computed in the algorithm earlier.
         */
        private transient double xD = Double.NaN;

        /**
         * {@inheritDoc}This method checks for suitability of non linear
         * interpolator to be used given the three points that fit a parabola.
         * However it may switch to linear interpolator if nonlinear estimates
         * are bad. The interpolate method computes the interpolated value for a
         * given value of x={@link #xD} which is already set during
         * construction.
         *
         * @return an interpolating polynomial function
         * @throws NotANumberException in case if xD is set as
         *             {@link Double#NaN}
         * @throws NullArgumentException in case of null arrays
         * @throws DimensionMismatchException in case xval[] and yval[] have
         *             differing lengths
         */
        @SuppressWarnings("serial")
        public UnivariateFunction interpolate(double[] xval, double[] yval) {
            if (Double.isNaN(xD)) {
                throw new NotANumberException() {
                    {
                        getContext().addMessage(
                                LocalizedFormats.NAN_NOT_ALLOWED);
                    }
                };
            }
            if (xval == null || yval == null) {
                throw new NullArgumentException(
                        LocalizedFormats.ARRAY_ZERO_LENGTH_OR_NULL_NOT_ALLOWED);
            }
            if (xval.length < 3 || xval.length != yval.length) {
                throw new DimensionMismatchException(
                      LocalizedFormats.INSUFFICIENT_DIMENSION, xval.length, 3);
            }

            double[] x = xval;
            double[] y = yval;
            UnivariateFunction univariateFunction = nonLinear.interpolate(x, y);
            double yD = univariateFunction.value(xD);

            if (isEstimateBad(y, yD)) {
                int d = xD - x[1] > 0 ? 1 : -1;
                x = new double[] { x[1], x[1 + d] };
                y = new double[] { y[1], y[1 + d] };
                MathArrays.sortInPlace(x, y);// since d can be +/- 1
                univariateFunction = linear.interpolate(x, y);
                linearEstimationCount++;
            } else {
                quadraticEstimationCount++;
            }
            return univariateFunction;
        }

        /**
         * Check if parabolic/nonlinear estimate is bad by checking if the
         * ordinate found is beyond the y[0] and y[2].
         *
         * @param y the array to get the bounds
         * @param yD the estimate
         * @return true if yD is a bad estimate
         */
        private boolean isEstimateBad(double[] y, double yD) {
            return yD <= y[0] || yD >= y[2];
        }

        /**
         * {@inheritDoc}
         */
        public double estimate(double[] x, double[] y, double z) {
            xD = z;
            return interpolate(x, y).value(z);
        }

        /**
         * {@inheritDoc}
         */
        public int quadraticEstimationCount() {
            return quadraticEstimationCount;
        }

        /**
         * {@inheritDoc}
         */
        public int linearEstimationCount() {
            return linearEstimationCount;
        }

        /**
         * Sets xD
         *
         * @param xDesired desired x position
         */
        public void xD(double xDesired) {
            this.xD = xDesired;
        }

        /**
         * This method is used to set up some post construction initializations.
         *
         * @param anInputStream input stream to read from
         * @throws ClassNotFoundException thrown when a class is absent for
         *             deserializing
         * @throws IOException thrown when IO Error occurs
         */
        private void readObject(ObjectInputStream anInputStream)
                throws ClassNotFoundException, IOException {
            anInputStream.defaultReadObject();
            linear = new LinearInterpolator();
            nonLinear = new NevilleInterpolator();
            xD = Double.NaN;
            linearEstimationCount = 0;
            quadraticEstimationCount = 0;
        }

        @Override
        public boolean equals(Object o) {
            boolean result = o != null && getClass().isAssignableFrom(
                    o.getClass());
            if (result) {
                PSquareInterpolatorEvaluator e =
                        (PSquareInterpolatorEvaluator) o;
                boolean isLinearEqual =
                        linearEstimationCount == e.linearEstimationCount;
                boolean isQuadraticEqual =
                        quadraticEstimationCount == e.quadraticEstimationCount;
                result = isLinearEqual  && isQuadraticEqual;
            }
            return result;
        }

        @Override
        public int hashCode() {
            return getClass().hashCode() + linear.getClass().hashCode();
        }
    }

    /**
     * A simple fixed capacity list that has an upper bound to grow till
     * capacity. This class is private as it has very specific purpose of
     * bounding the capacity required for pSquare algorithm.. TODO:
     * AbstractSerializablListDecorator of commons collection could be used as a
     * base instead and whenever pom.xml of math3 allows adding the same.
     * <p>
     * This class is private static as it its specifically needed for this
     * algoritm to work
     *
     * @param <E>
     */
    private static class FixedCapacityList<E> extends ArrayList<E>
    implements Serializable {
        /**
         * Serialization Version Id
         */
        private static final long serialVersionUID = 2283952083075725479L;
        /**
         * Capacity of the list
         */
        private final int capacity;

        /**
         * This constructor constructs the list with given capacity and as well
         * as stores the capacity
         *
         * @param fixedCapacity the capacity to be fixed for this list
         */
        public FixedCapacityList(int fixedCapacity) {
            super(fixedCapacity);
            this.capacity = fixedCapacity;
        }

        /**
         * {@inheritDoc} In addition it checks if the {@link #size()} returns a
         * size that is within {@link #capacity} and if true it adds.
         *
         * @return true if addition is successful and false otherwise
         */
        @Override
        public boolean add(E e) {
            return size() < capacity() ? super.add(e) : false;
        }

        /**
         * {@inheritDoc} In addition it checks if the sum of Collection size and
         * this instance's {@link #size()} returns a value that is within
         * {@link #capacity} and if true it adds the collection.
         *
         * @return true if addition is successful and false otherwise
         */
        @Override
        public boolean addAll(Collection<? extends E> collection) {
            boolean isCollectionLess =
                   collection != null && collection.size() + size() <= capacity;
            return isCollectionLess ? super.addAll(collection) : false;
        }

        /**
         * Return the capacity set during construction
         *
         * @return capacity of list
         */
        public int capacity() {
            return capacity;
        }
    }
}
