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
import org.apache.commons.math3.analysis.function.StepFunction;
import org.apache.commons.math3.analysis.interpolation.LinearInterpolator;
import org.apache.commons.math3.analysis.interpolation.NevilleInterpolator;
import org.apache.commons.math3.analysis.interpolation.UnivariateInterpolator;
import org.apache.commons.math3.exception.MathIllegalArgumentException;
import org.apache.commons.math3.exception.MathIllegalStateException;
import org.apache.commons.math3.exception.NotANumberException;
import org.apache.commons.math3.exception.OutOfRangeException;
import org.apache.commons.math3.exception.util.LocalizedFormats;
import org.apache.commons.math3.stat.descriptive.AbstractStorelessUnivariateStatistic;
import org.apache.commons.math3.stat.descriptive.StorelessUnivariateStatistic;
import org.apache.commons.math3.util.MathArrays;
import org.apache.commons.math3.util.Precision;

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
 * <b>Note: This implementation is not synchronized</b>
 * 
 * @author vmurthy
 * 
 */

public class PSquaredPercentile extends AbstractStorelessUnivariateStatistic
		implements StorelessUnivariateStatistic, Serializable {
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
	 * Integer Formatter
	 */
	private static final DecimalFormat intFormat = new DecimalFormat("00");

	/**
	 * A Decimal format for printing convenience
	 */
	private static final DecimalFormat df = new DecimalFormat("00.00");

	/**
	 * Initial list of 5 numbers corresponding to 5 markers.
	 * <p>
	 * Please watch out for the add methods that are overloaded
	 * <p>
	 * TODO: I may need to replace this with a wrapper using
	 * AbstractSerializableListDecorator(I cannot use FixedList here)
	 */
	private List<Double> initialFive = new FixedCapacityList<Double>(5);

	/**
	 * The quantile needed should be in range of 0-1 or 1-100
	 */
	private final double quantile;

	/**
	 * lastObservation is the last observation value/input sample.
	 * <p>
	 * no need to serialize
	 */
	transient private double lastObservation;

	/**
	 * The {@link PSquareEstimator} of Quantile. TODO : use strategy here
	 * <p>
	 * No need to serialize
	 */
	transient private PSquareEstimator estimator = new PSquareInterpolatorEvaluator();
	// new PiecewisePSquareInterpolatorEvaluator();

	/**
	 * {@link Markers} is the Marker Collection object which comes to effect
	 * only after 5 values are inserted
	 */
	private Markers markers = null;

	/**
	 * Computed p value (i,e percentile value of data set hither to received)
	 */
	private double pValue = Double.NaN;

	/**
	 * Counter to count the values accepted into this data set
	 */
	private long N;

	/**
	 * {@inheritDoc} and any other attributes.
	 */
	public int hashCode() {
		double result = getResult();
		result = Double.isNaN(result) ? 37 : result;
		result = result + quantile;// ((markers != null) ? markers.hashCode() :
									// 47);
		return (int) (result * 31 + getN() * 13);
	}

	/**
	 * {@inheritDoc};However in addition in this class a check on the equality
	 * of {@link Markers} is also made.While checking for results NaNs are
	 * appropriately considered
	 */
	public boolean equals(Object o) {
		if (this == o)
			return true;
		if (o != null && o instanceof PSquaredPercentile) {
			PSquaredPercentile that = (PSquaredPercentile) o;
			boolean result = markers != null && that.markers != null ? markers
					.equals(that.markers) : markers == null
					&& that.markers == null; // Its possible to have null
												// markers as in the case first
												// five observations
			return result
					&& getN() == that.getN()
					&& (!Double.isNaN(getResult())
							&& !Double.isNaN(that.getResult()) ? getResult() == that
							.getResult() : (Double.isNaN(getResult()) && Double
							.isNaN(that.getResult())));
		} else
			return false;
	}

	/**
	 * This method is used to set up some post construction initializations.
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
	 * Constructor with passed in percentile. If the value passed is within
	 * [0,1] then its considered as is but if it is (1-100] then it is divided
	 * by 100
	 * 
	 * @param percentile
	 *            needed and should be within [0-100]
	 * @throws OutOfRangeException
	 *             in case of percentile being asked is NOT within [0-100]
	 */
	public PSquaredPercentile(Number percentile) {
		if ((percentile.intValue() > 100) || (percentile.intValue() < 0)) {
			throw new OutOfRangeException(
					LocalizedFormats.OUT_OF_BOUNDS_QUANTILE_VALUE,
					percentile.intValue(), 0, 100);
		}
		this.quantile = (percentile.intValue() > 1) ? percentile.intValue() / 100.0
				: percentile.doubleValue();

	}

	/**
	 * Default constructor that assumes a {@link #DEFAULT_QUANTILE_DESIRED
	 * default quantile} needed
	 */
	PSquaredPercentile() {
		this(DEFAULT_QUANTILE_DESIRED);
	}

	/**
	 * {@inheritDoc}The internal state updated due to the new value in this
	 * context is basically of the <a
	 * href=http://www.cs.wustl.edu/~jain/papers/ftp/psqr.pdf>P<sup>2</sup></a>
	 * algorithm marker positions and computation of the approximate quantile.
	 * <p>
	 * Increments/Accept a value / observation into the data set and computes
	 * percentile. The result always must be queried using {@link #getResult()}
	 * 
	 * @param observation
	 *            is the observation currently being added.
	 * 
	 */
	public void increment(final double observation) {
		// Increment counter
		N++;

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
		pValue = markers.processADataPoint(observation);
	}

	/**
	 * {@inheritDoc}. This populates a string containing the values of the
	 * attributes all {@link Marker}s. Also adds a bit of formatting with
	 * separators for convenience view.
	 */
	public String toString() {

		if (markers == null)
			return String
					.format("|%s |%s|%s|---------------|------------------------------|-----------------------------|",
							df.format(lastObservation), "-", df.format(pValue));
		else
			return String.format("|%s %s", df.format(lastObservation),
					markers.toString());
	}

	/**
	 * {@inheritDoc}
	 * 
	 * @see org.apache.commons.math3.stat.descriptive.StorelessUnivariateStatistic
	 *      #getN()
	 */
	public long getN() {
		return N;
	}

	/**
	 * {@inheritDoc}
	 * 
	 * @see org.apache.commons.math3.stat.descriptive.AbstractStorelessUnivariateStatistic#copy()
	 */
	@Override public StorelessUnivariateStatistic copy() {
		PSquaredPercentile copy;

		if (markers != null) {

			Markers marks = new Markers(initialFive, this.quantile())
					.initialize(markers.m.clone()).estimator(estimator);
			marks.postConstruct();
			copy = new PSquaredPercentile(quantile).estimator(estimator)
					.markers(marks);
		} else
			copy = new PSquaredPercentile(quantile).estimator(estimator);
		copy.N = N;
		copy.pValue = pValue;
		// TODO: I may need to replace this with a wrapper using
		// AbstractSerializableListDecorator (I cannot use FixedList here)
		copy.initialFive = new FixedCapacityList<Double>(5);
		copy.initialFive.addAll(initialFive);
		return copy;
	}

	/**
	 * Sets the estimator
	 * 
	 * @return this instance
	 */
	PSquaredPercentile estimator(PSquareEstimator estimator) {
		this.estimator = estimator;
		return this;
	}

	/**
	 * Sets the {@link Markers}
	 * 
	 * @param markers
	 * @return this
	 */
	PSquaredPercentile markers(Markers markers) {
		this.markers = markers.estimator(estimator);
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
	 * {@link #initialFive} list and {@link #N} to 0
	 * 
	 * @see org.apache.commons.math3.stat.descriptive.AbstractStorelessUnivariateStatistic#clear()
	 */
	@Override public void clear() {
		markers = null;
		initialFive.clear();
		N = 0L;
		pValue = Double.NaN;
	}

	/**
	 * {@inheritDoc} This is basically the computed quantile value stored in
	 * pValue.
	 * 
	 * @see org.apache.commons.math3.stat.descriptive.AbstractStorelessUnivariateStatistic#getResult()
	 */
	@Override public double getResult() {
		if (quantile == 1)
			pValue = maximum();
		else if (quantile == 0)
			pValue = minimum();
		return pValue;
	}

	/**
	 * Return Maximum in case of quantile=1
	 * 
	 * @return maximum in the data set added to this statistic
	 */
	private double maximum() {
		double val = Double.NaN;
		if (markers != null)
			val = markers.m[5].q;
		else if (!initialFive.isEmpty())
			val = initialFive.get(initialFive.size() - 1);
		return val;
	}

	/**
	 * Return Minimum in case of quantile=0
	 * 
	 * @return minimum in the data set added to this statistic
	 */
	private double minimum() {
		double val = Double.NaN;
		if (markers != null)
			val = markers.m[1].q;
		else if (!initialFive.isEmpty())
			val = initialFive.get(0);
		return val;
	}

	/**
	 * The Markers is an encapsulation of the five markers/buckets as indicated
	 * in <a
	 * href=http://www.cs.wustl.edu/~jain/papers/ftp/psqr.pdf>P<sup>2</sup></a>
	 * algorithm.
	 * 
	 * @author vmurthy
	 * 
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
		 * algorithm paper</a>)
		 */
		private Marker[] m;
		/**
		 * Kth cell belonging to [1-5] of the {@link #m}. Dont need this to be
		 * serialized
		 */
		transient private int k = -1;
		/**
		 * The {@link PSquareEstimator} instance to be set every time. Not
		 * serializing this
		 */
		transient private PSquareEstimator estimator;

		/**
		 * This equals method basically checks for marker array to be deep
		 * equals and estimator to be of same class.
		 * 
		 * @param o
		 *            is the other object
		 * @return true if the passed in and this object are equivalent
		 */
		public boolean equals(Object o) {
			boolean result = false;
			if (this == o)
				result = true;
			else if (o != null && o instanceof Markers) {
				Markers that = (Markers) o;
				result = Arrays.deepEquals(m, that.m);
				result = result
						&& ((estimator != null && that.estimator != null) ? estimator
								.getClass().isAssignableFrom(
										that.estimator.getClass()) : true);
			}
			return result;
		}

		/**
		 * Accessor for getting array of {@link Marker} that is {@link #m}
		 * 
		 * @return array of {@link Marker} that is {@link #m}
		 */
		Marker[] m() {
			return m;
		}

		/**
		 * This basically calls the {@link #postConstruct()} to set up the
		 * linking indexes which is not captured while serializing.
		 * 
		 * @param aInputStream
		 * @throws ClassNotFoundException
		 * @throws IOException
		 */
		private void readObject(ObjectInputStream aInputStream)
				throws ClassNotFoundException, IOException {
			// always perform the default de-serialization first
			aInputStream.defaultReadObject();
			postConstruct();
		}

		/**
		 * Constructor
		 * 
		 * @param initial
		 *            is a list of first five values
		 * @param p
		 *            is the quantile needed
		 * @throws MathIllegalArgumentException
		 *             in case if initial list is null or having less than 5
		 *             elements
		 * 
		 */
		Markers(List<Double> initial, double p) {
			if (initial == null || initial.size() < 5) {
				int countObserved = initial == null ? -1 : initial.size();
				throw new MathIllegalArgumentException(
						LocalizedFormats.INSUFFICIENT_OBSERVED_POINTS_IN_SAMPLE,
						countObserved, 5);
			}
			Collections.sort(initial);
			initialize(m = new Marker[] {
					new Marker(),// Null Marker
					new Marker(initial.get(0), 1, 0, 1),
					new Marker(initial.get(1), 1 + 2 * p, p / 2, 2),
					new Marker(initial.get(2), 1 + 4 * p, p, 3),
					new Marker(initial.get(3), 3 + 2 * p, (1 + p) / 2, 4),
					new Marker(initial.get(4), 5, 1, 5) });
		}

		/**
		 * A Setter for setting the {@link PSquareEstimator} which can be one of
		 * {@link PSquareInterpolatorEvaluator} or
		 * {@link PiecewisePSquareInterpolatorEvaluator} as of the moment
		 * 
		 * @param estimator
		 *            is {@link PSquareEstimator} passed in
		 * @return this instance
		 */
		Markers estimator(PSquareEstimator estimator) {
			this.estimator = estimator;
			return this;
		}

		/**
		 * Initialize markers with the passed {@link Marker} array
		 * 
		 * @param markers
		 *            passed in array of {@link Marker}
		 * @return this instance
		 */
		Markers initialize(Marker[] markers) {
			for (int i = 0; i < markers.length; i++)
				m[i].initialize(markers[i]);
			postConstruct();
			return this;
		}

		/**
		 * A Post construct method which builds the links and initializes marker
		 * indexes. It also sets the estimator. 
		 * <p>TODO: The estimator needs to be taken from a Strategy Pattern
		 */

		void postConstruct() {
			assert m.length == 5 + 1;
			for (int i = 1; i < 5; i++)
				m[i].previous(m[i - 1]).next(m[i + 1]).index(i);
			m[0].previous(m[0]).next(m[1]).index(0);
			m[5].previous(m[4]).next(m[5]).index(5);
			if (estimator == null)
				estimator = new PSquareInterpolatorEvaluator();
		}

		/**
		 * Process a Data point
		 * 
		 * @param x
		 * @return computed percentile
		 */
		public double processADataPoint(double x) {

			// 1. Find cell and update minima and maxima
			int kthCell = findCellAndUpdateMinMax(x);

			// 2. Increment positions
			incPos(1, kthCell + 1, 5);

			// 2a. Update desired position with increments
			updateDesiredPos();

			// 3. Adjust heights of m[2-4] if necessary
			adjustHeightsOfMarkers(estimator);

			// 4. Return percentile
			return getPValue();
		}

		/**
		 * @return pValue which is mid point
		 */
		public double getPValue() {
			return m[3].q;
		}

		/**
		 * Finds the cell where x fits
		 * <p>
		 * TODO: Check if a sort of {@link StepFunction} can be used (Ofcourse
		 * StepFunction cannot exactly fit here)
		 * 
		 * @param observation
		 *            is the observation
		 * @return kth cell (of the markers ranging from 1-5) where observed
		 *         sample fits
		 * @throws MathIllegalStateException
		 *             in case if markers are not 6 (1+5) or if cell index is
		 *             beyond bounds [1-5]
		 */
		private int findCellAndUpdateMinMax(double observation) {
			assert m.length == 5 + 1; // this works only if markers are 5
			k = -1;
			if (observation < m[1].q) {
				m[1].q = observation;
				k = 1;
			} else if (observation < m[2].q)
				k = 1;
			else if (observation < m[3].q)
				k = 2;
			else if (observation < m[4].q)
				k = 3;
			else if (observation <= m[5].q)
				k = 4;
			else {
				m[5].q = observation;
				k = 4;
			}
			return k;
		}

		/**
		 * Adjust marker heights by setting quantile estimates to middle markers
		 */
		private void adjustHeightsOfMarkers(PSquareEstimator estimator) {
			for (int i = 2; i <= 4; i++)
				m[i].estimate(estimator);
		}

		/**
		 * Increment positions by d. Please refer to algorithm paper for concept
		 * of d
		 * 
		 * @param d
		 * @param startIndex
		 * @param endIndex
		 */
		private void incPos(int d, int startIndex, int endIndex) {
			for (int i = startIndex; i <= endIndex; i++)
				m[i].incPos(d);
		}

		/**
		 * Desired positions incremented by bucket width. bucket width is
		 * basically the desired increments
		 */
		private void updateDesiredPos() {
			for (int i = 1; i < m.length; i++)
				m[i].updateDesiredPos();
		}

		/**
		 * toString
		 */
		public String toString() {

			return String.format("|%d|%s|%s|%s|%s|", k, df.format(getPValue()),
					doubleToString(ns(), intFormat, " "),
					doubleToString(qs(), df, " "),
					doubleToString(nps(), df, " "));
		}

		/**
		 * doubleToString provides a formatted representation of double array
		 * 
		 * @param doubleArray
		 *            passed in array
		 * @param decimalFormat
		 *            is the formatter
		 * @param delimiter
		 *            a delimiter between elements
		 * @return String representation of array
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
		 * @return an array of all of {@link Marker#n}s
		 */
		public Double[] ns() {

			return new Double[] { Precision.round(m[1].n, 0),
					Precision.round(m[2].n, 0), Precision.round(m[3].n, 0),
					Precision.round(m[4].n, 0), Precision.round(m[5].n, 0) };
		}

		/**
		 * The desired positions nps
		 * 
		 * @return an array of all of {@link Marker#np}s
		 */
		public Double[] nps() {
			return new Double[] { Precision.round(m[1].np, 2),
					Precision.round(m[2].np, 2), Precision.round(m[3].np, 2),
					Precision.round(m[4].np, 2), Precision.round(m[5].np, 2) };
		}

		/**
		 * The quantile array of all the markers
		 * 
		 * @return an array of all of {@link Marker#q}s
		 */
		public Double[] qs() {
			return new Double[] { Precision.round(m[1].q, 2),
					Precision.round(m[2].q, 2), Precision.round(m[3].q, 2),
					Precision.round(m[4].q, 2), Precision.round(m[5].q, 2) };
		}

		/**
		 * Returns the Markers as a Map
		 * 
		 * @return Map of marker attributes
		 */
		public Map<String, Object> toMap() {
			Map<String, Object> map = new LinkedHashMap<String, Object>();
			for (Marker mark : m)
				map.putAll(mark.toMap());
			return map;
		}

	}

	/**
	 * 
	 * The class modeling the attributes of The Marker of the <a
	 * href=http://www.cs.wustl.edu/~jain/papers/ftp/psqr.pdf>P<sup>2</sup></a>
	 * algorithm. Keeping the same variable names as is indicated in the <a
	 * href=http://www.cs.wustl.edu/~jain/papers/ftp/psqr.pdf>P<sup>2</sup>
	 * algorithm paper</a>.
	 * 
	 * @author vmurthy
	 * 
	 */

	protected static class Marker implements Serializable {

		/**
		 * Serial Version ID
		 */
		private static final long serialVersionUID = -3575879478288538431L;

		/**
		 * The Marker Index which is just a serial number for the marker
		 */
		private int index;

		/**
		 * n the integral marker position. Refer to the <a
		 * href=http://www.cs.wustl.edu/~jain/papers/ftp/psqr.pdf>P<sup>2</sup>
		 * algorithm paper</a> for the variable n
		 */
		private double n;

		/**
		 * Desired Marker Position. Refer to the <a
		 * href=http://www.cs.wustl.edu/~jain/papers/ftp/psqr.pdf>P<sup>2</sup>
		 * algorithm paper</a> for the variable n'
		 */
		private double np;

		/**
		 * Marker height or the quantile. Refer to the <a
		 * href=http://www.cs.wustl.edu/~jain/papers/ftp/psqr.pdf>P<sup>2</sup>
		 * algorithm paper</a> for the variable q
		 */
		private double q;

		/**
		 * Desired Marker increment. Refer to the <a
		 * href=http://www.cs.wustl.edu/~jain/papers/ftp/psqr.pdf>P<sup>2</sup>
		 * algorithm paper</a> for the variable dn'
		 */
		private double dn;

		/**
		 * Next and Previous Markers for easy linked navigation in loops. this
		 * is not serialized as they can be rebuilt during de-serialization.
		 */
		transient private Marker next, previous;

		/**
		 * Constructor
		 * 
		 * @param q
		 *            Refer to javadoc of {@link #q}
		 * @param np
		 *            Refer to javadoc of {@link #np}
		 * @param dn
		 *            Refer to javadoc of {@link #dn}
		 * @param n
		 *            Refer to javadoc of {@link #n}
		 */
		public Marker(double q, double np, double dn, double n) {
			this.q = q;
			this.np = np;
			this.dn = dn;
			this.n = n;
			this.next = this.previous = this;
		}

		/**
		 * Default constructor
		 */
		public Marker() {
		}

		/**
		 * Navigates to the next marker
		 * 
		 * @return the next marker set
		 */
		Marker next() {
			return next;
		}

		/**
		 * Navigates to the previous marker
		 * 
		 * @return previous marker set
		 */
		Marker previous() {
			return previous;
		}

		/**
		 * Sets the previous marker
		 * 
		 * @param marker
		 * @return this instance
		 */
		Marker previous(Marker marker) {
			this.previous = marker;
			return this;
		}

		/**
		 * Sets the next marker
		 * 
		 * @param marker
		 * @return this instance
		 */
		Marker next(Marker marker) {
			this.next = marker;
			return this;
		}

		/**
		 * Sets the index of the marker
		 * 
		 * @param index
		 * @return this instance
		 */
		Marker index(int index) {
			this.index = index;
			return this;
		}

		/**
		 * Initializes with passed in Marker / Copy ctor. No changes to previous
		 * and next pointers.
		 * 
		 * @param marker
		 *            passed in
		 */
		Marker initialize(Marker marker) {
			return q(marker.q).np(marker.np).n(marker.n).dn(marker.dn);
		}

		/**
		 * Sets quantile q to this instance
		 * 
		 * @param q
		 *            is the quantile to be set
		 * @return this instance
		 */
		Marker q(double q) {
			this.q = q;
			return this;
		}

		/**
		 * Sets desired position np ( a double quantity) of this marker
		 * 
		 * @param np
		 *            is the desired position ( a real number) to be set
		 * @return this instance
		 */
		Marker np(double np) {
			this.np = np;
			return this;
		}

		/**
		 * Sets integral position n of this marker
		 * 
		 * @param n
		 *            is the integral numbered position
		 * @return this instance
		 */
		Marker n(double n) {
			this.n = n;
			return this;
		}

		/**
		 * Sets the desired increments
		 * 
		 * @param dn
		 *            is the desired increment
		 * @return this instance
		 */
		Marker dn(double dn) {
			this.dn = dn;
			return this;
		}

		/**
		 * Update desired Position with increment
		 */
		void updateDesiredPos() {
			np += dn;
		}

		/**
		 * Increment Position by d
		 * 
		 * @param d
		 *            value to increment
		 */
		void incPos(int d) {
			n += d;
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
		 * @return Difference between desired and actual position
		 */
		double diff() {
			return np - n;
		}

		/**
		 * Builds a Map of attribute names and their values and returns
		 * 
		 * @return map containing keys (index, n ,np, q, dn with each of these
		 *         prefixed with {@link #name()}) and their corresponding values
		 *         contained in this instance
		 */
		public Map<String, Object> toMap() {
			Map<String, Object> map = new LinkedHashMap<String, Object>();
			map.put(name() + ".index", (double) index);
			map.put(name() + ".n", Precision.round(n, 0));
			map.put(name() + ".np", Precision.round(np, 2));
			map.put(name() + ".q", Precision.round(q, 2));
			map.put(name() + ".dn", Precision.round(dn, 2));
			return map;
		}

		/**
		 * {@inheritDoc}. This class basically returns a {@link #toMap() map
		 * view} of the attributes in a string form
		 */
		public String toString() {
			return toMap().toString();
		}

		/**
		 * estimate the quantile for the current marker
		 * 
		 * @param estimator
		 *            is an instance of {@link PSquareEstimator} passed in
		 * @return estimated quantile
		 */
		public double estimate(PSquareEstimator estimator) {
			double di = diff();
			if (((di >= 1) && ((next.n - n) > 1))
					|| ((di <= -1) && ((previous.n - n) < -1))) {
				int d = di >= 0 ? 1 : -1;
				double[] x = new double[] { previous.n, n, next.n };
				double[] y = new double[] { previous.q, q, next.q };
				q = estimator.estimate(x, y, n + d);
				incPos(d);
			}
			return q;
		}

		/**
		 * {@inheritDoc}<i>This equals method checks for equivalence of
		 * {@link #n},{@link #np},{@link #dn} and {@link #q} and in addition
		 * checks if navigation pointers({@link #next} and {@link #previous})
		 * are the same between this and passed in object</i>
		 * 
		 * @param o
		 *            Other object
		 * @return true if this equals passed in other object o
		 */
		public boolean equals(Object o) {
			boolean result = false;
			if (this == o)
				result = true;
			else if (o != null && o instanceof Marker) {
				Marker that = (Marker) o;
				result = q == that.q && n == that.n && np == that.np
						&& dn == that.dn;
				result = result
						&& ((next == null || that.next == null) ? true
								: next.index == that.next.index)
						&& ((previous == null || that.previous == null) ? true
								: previous.index == that.previous.index);
			}
			return result;
		}
	}

	/**
	 * An Interface modeling to estimate quantile based on <a
	 * href=http://www.cs.wustl.edu/~jain/papers/ftp/psqr.pdf>P<sup>2</sup>
	 * algorithm</a> which require 3 points to fit a parabolic curve.
	 * <p>
	 * It also makes use of polynomial interpolation to find an estimate of the
	 * quantile;however switches to a linear interpolation in case if polynomial
	 * interpolation results in an undesirable results.Please refer to details
	 * in the <a
	 * href=http://www.cs.wustl.edu/~jain/papers/ftp/psqr.pdf>P<sup>2</sup>
	 * algorithm paper</a> as to when the switch happens to linear
	 * interpolation.
	 * <p>
	 * The reason for making this as interface is to allow for future usage on
	 * different possibilities of polynomial interpolations
	 * 
	 * @author vmurthy
	 * 
	 */
	protected interface PSquareEstimator {
		/**
		 * Estimates the quantile based on passed in x and y array and a point
		 * z.
		 * 
		 * @param x
		 *            is a 3 element array corresponding to x coordinate of the
		 *            three points (x[0], x[1], x[2])
		 * @param y
		 *            is a 3 element array corresponding to y coordinate of the
		 *            three points (y[0], y[1], y[2])
		 * @param z
		 *            is a x value off the value of x[1] by a distance d
		 * @return the P<sup>2</sup> estimation value of quantile at a value z
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
	 * This is the classic/traditional procedural style implementation of the <a
	 * href=http://www.cs.wustl.edu/~jain/papers/ftp/psqr.pdf>P<SUP>2</SUP>
	 * Algorithm</a> Estimation as is explained in the original works.
	 * <p>
	 * The difference between this class and
	 * {@link PSquareInterpolatorEvaluator} is basically that this class does'nt
	 * make use of any of the existing {@link UnivariateInterpolator}s.
	 * <p>
	 * Provided this as reference to cross verify
	 * 
	 * @author vmurthy
	 * 
	 */
	protected static class PiecewisePSquareInterpolatorEvaluator implements
			PSquareEstimator {
		int qCount, lCount = 0;

		/**
		 * {@inheritDoc}This method computes the quantile in the old classic way
		 * as is given in the <a
		 * href=http://www.cs.wustl.edu/~jain/papers/ftp/psqr.pdf>P<SUP>2</SUP>
		 * Algorithm Paper</a>
		 */
		public double estimate(double[] x, double[] y, double z) {
			if (Double.isNaN(z))
				throw new NotANumberException() {
					{
						getContext().addMessage(
								LocalizedFormats.NAN_ELEMENT_AT_INDEX, "z");
					}
				};
			if (x == null || y == null)
				throw new MathIllegalArgumentException(
						LocalizedFormats.ARRAY_ZERO_LENGTH_OR_NULL_NOT_ALLOWED);
			if (x.length < 3 || x.length != y.length)
				throw new MathIllegalArgumentException(
						LocalizedFormats.INSUFFICIENT_DIMENSION, x.length, 3);
			double m2 = (y[2] - y[1]) / (x[2] - x[1]);
			double m1 = (y[1] - y[0]) / (x[1] - x[0]);
			int d = (z - x[1]) > 0 ? 1 : -1;
			// Quadratic - Second order
			double qip = y[1] + (d / (x[2] - x[0]))
					* ((x[1] - x[0] + d) * m2 + (x[2] - x[1] - d) * m1);
			// Linear - First order
			if (y[0] >= qip || qip >= y[2]) {
				qip = y[1] + d * (y[1 + d] - y[1]) / (x[1 + d] - x[1]);
				lCount++;
			} else
				qCount++;
			return qip;
		}

		/**
		 * {@inheritDoc}
		 */
		public int quadraticEstimationCount() {
			return qCount;
		}

		/**
		 * {@inheritDoc}
		 */
		public int linearEstimationCount() {
			return lCount;
		}
	}

	/**
	 * This is a Modular Object Oriented Implementation of <a
	 * href=http://www.cs.wustl.edu/~jain/papers/ftp/psqr.pdf>P<SUP>2</SUP>
	 * Algorithm</a> estimation by delegating to existing instances of
	 * {@link UnivariateInterpolator} such as {@link NevilleInterpolator} for
	 * the first estimation based on quadratic formulae and
	 * {@link LinearInterpolator} in case if needed for correction as is
	 * reasoned out in the <a
	 * href=http://www.cs.wustl.edu/~jain/papers/ftp/psqr.pdf>P<SUP>2</SUP>
	 * Algorithm</a> for the choice of quadratic vs linear estimation
	 * 
	 * @author vmurthy
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
		 * <p>
		 * No need for serializing
		 */
		transient final UnivariateInterpolator nonLinear = new NevilleInterpolator();// ParabolicInterpolator();
		/**
		 * A linear interpolator used to correct in case if non linear
		 * interpolated gives an out of bounds.
		 * <p>
		 * No need for serializing
		 */
		transient final UnivariateInterpolator linear = new LinearInterpolator();
		/**
		 * lCount counts the number of times for linear estimation used.
		 * <p>
		 * No Need for serializing
		 */
		transient int lCount = 0;
		/**
		 * qCount counts the number of times for non linear estimation used.(q
		 * actually means quadratic).
		 * <p>
		 * No need for serializing
		 */
		transient int qCount = 0;
		/**
		 * xD is the desired xth position for which quantile needs to be
		 * calculated.
		 * <p>
		 * No need for serializing
		 */
		transient double xD = Double.NaN;

		/**
		 * {@inheritDoc}.
		 * <p>
		 * This method checks for suitability of non linear interpolator to be
		 * used given the three points that fit a parabola.However switches to
		 * linear interpolator if estimates are bad.
		 * 
		 * @return An Interpolating Polynomial Function
		 * @throws NotANumberException
		 *             in case if xD is not set other than {@link Double#NaN}
		 * @throws MathIllegalArgumentException
		 *             in case zero array length
		 */
		@SuppressWarnings("serial") public UnivariateFunction interpolate(
				double[] xval, double[] yval) {
			if (Double.isNaN(xD))
				throw new NotANumberException() {
					{
						getContext().addMessage(
								LocalizedFormats.NAN_ELEMENT_AT_INDEX, "xD");
					}
				};
			if (xval == null || yval == null)
				throw new MathIllegalArgumentException(
						LocalizedFormats.ARRAY_ZERO_LENGTH_OR_NULL_NOT_ALLOWED);
			if (xval.length < 3 || xval.length != yval.length)
				throw new MathIllegalArgumentException(
						LocalizedFormats.INSUFFICIENT_DIMENSION, xval.length, 3);

			double[] x = xval, y = yval;
			UnivariateFunction univariateFunction = nonLinear.interpolate(x, y);
			double yD = univariateFunction.value(xD);

			if (isEstimateBad(y, yD)) {
				int d = (xD - x[1]) > 0 ? 1 : -1;
				x = new double[] { x[1], x[1 + d] };
				y = new double[] { y[1], y[1 + d] };
				MathArrays.sortInPlace(x, y);// since d can be +/- 1
				univariateFunction = linear.interpolate(x, y);
				lCount++;
				System.out
						.format("\nLinear:xD=%f,qip=%f,d=%d,x[0]=%f,x[1]=%f,x[2]=%f,y[0]=%f,y[1]=%f,y[2]=%f",
								xD, univariateFunction.value(xD), d, xval[0],
								xval[1], xval[2], yval[0], yval[1], yval[2]);
			} else {
				qCount++;
			}
			return univariateFunction;
		}

		/**
		 * Check if parabolic/nonlinear estimate is bad by checking if the
		 * ordinate found is beyond the y[0] and y[2].
		 * 
		 * @param y
		 *            is the array to get the bounds
		 * @param yD
		 *            is the estimate
		 * @return true if its bad estimate
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
			return qCount;
		}

		/**
		 * {@inheritDoc}
		 */
		public int linearEstimationCount() {
			return lCount;
		}
	}

	/**
	 * A Simple fixed capacity list that has an upper bound to grow till
	 * capacity. <p>I could have actually used AbstractSerializablListDecorator of
	 * commons collection;however since i cant add commons-collection into pom.xml; needed
	 * this class
	 * 
	 * @author vmurthy
	 * 
	 * @param <E>
	 */
	protected static class FixedCapacityList<E> extends ArrayList<E> implements
			Serializable {
		private static final long serialVersionUID = 2283952083075725479L;
		/**
		 * Capacity of the list
		 */
		private final int capacity;

		/**
		 * {@inheritDoc}.This constructor constructs the list with given
		 * capacity and as well as stores the capacity
		 * 
		 * @param capacity
		 */
		public FixedCapacityList(int capacity) {
			super(capacity);
			this.capacity = capacity;
		}

		/**
		 * {@inheritDoc}This constructor populates collection c and sets
		 * {@link #capacity} as size of collection passed
		 * 
		 * @param c
		 */
		public FixedCapacityList(Collection<? extends E> c) {
			super(c);
			this.capacity = c.size();
		}

		/**
		 * {@inheritDoc} In addition it checks if the {@link #size()} returns a
		 * size that is within {@link #capacity} and if true it adds.
		 * 
		 * @return true if addition is successful and false otherwise
		 */
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
		public boolean addAll(Collection<? extends E> c) {
			return (c != null && (c.size() + size() <= capacity)) ? super
					.addAll(c) : false;
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
