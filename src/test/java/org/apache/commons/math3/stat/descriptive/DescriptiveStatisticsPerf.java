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

package org.apache.commons.math3.stat.descriptive;

import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.LockSupport;

import org.apache.commons.math3.PerfTestUtils;
import org.junit.Test;
/**
 * A simple performance test for various Descriptive Statistical Summary
 * <p>
 * <b>Hint:<code> mvn  clean test -Dtest=DescriptiveStatisticsPerf</code></b>
 */
public class DescriptiveStatisticsPerf {
	private static final int repeatStats = 100000;
	private static final AtomicInteger indexer = new AtomicInteger(1);
	static double[] dataArray=new double[repeatStats*2];
	static{
		Random r = new Random(Long.MAX_VALUE);
		for(int i=0;i<dataArray.length;i++)
			dataArray[i]=r.nextDouble();
	}
	static class Adder<S extends UnivariateStatistic> extends
			PerfTestUtils.RunTest {
		private final DescriptiveStatisticalSummary<S> stats;
		private final AtomicInteger randDataIndexer=new AtomicInteger(0);
		
		public Adder(DescriptiveStatisticalSummary<S> stats) {
			super(Adder.class.getSimpleName() + "-"
					+ stats.getClass().getSimpleName() + "-"
					+ indexer.getAndIncrement());
			this.stats = stats;
		}

		@Override
		public Double call() throws Exception {
			double d = dataArray[randDataIndexer.getAndIncrement()];
			stats.addValue(d);
			return d;
		}
	}

	static class Mean<S extends UnivariateStatistic> extends
			PerfTestUtils.RunTest {
		private final DescriptiveStatisticalSummary<S> stats;

		public Mean(DescriptiveStatisticalSummary<S> stats) {
			super(Mean.class.getSimpleName() + "-"
					+ stats.getClass().getSimpleName() + "-"
					+ indexer.getAndIncrement());
			this.stats = stats;
		}

		@Override
		public Double call() throws Exception {
			return stats.getMean();
		}
	}

	static class StdDeviation<S extends UnivariateStatistic> extends
			PerfTestUtils.RunTest {
		private final DescriptiveStatisticalSummary<S> stats;

		public StdDeviation(DescriptiveStatisticalSummary<S> stats) {
			super(StdDeviation.class.getSimpleName() + "-"
					+ stats.getClass().getSimpleName() + "-"
					+ indexer.getAndIncrement());
			this.stats = stats;
		}

		@Override
		public Double call() throws Exception {
			return stats.getStandardDeviation();
		}
	}

	static class Quantile<S extends UnivariateStatistic> extends
			PerfTestUtils.RunTest {
		private final DescriptiveStatisticalSummary<S> stats;

		public Quantile(DescriptiveStatisticalSummary<S> stats) {
			super(Quantile.class.getSimpleName() + "-"
					+ stats.getClass().getSimpleName() + "-"
					+ indexer.getAndIncrement());
			this.stats = stats;
		}

		@Override
		public Double call() throws Exception {
			return stats.getPercentile();
		}
	}

	@Test
	public void testAddLocked() {
		DescriptiveStatisticalSummary<StorelessUnivariateStatistic> locked = new LockedDescriptiveStorelessStatistics();
		StatisticalSummary[] summary = PerfTestUtils
				.timesAndResultsConcurrently(
						new PerfTestUtils.LongRunTest("AddLoc1",
								new Adder<StorelessUnivariateStatistic>(locked),
								repeatStats),
						new PerfTestUtils.LongRunTest("AddLoc2",
								new Adder<StorelessUnivariateStatistic>(locked),
								repeatStats),
						new PerfTestUtils.LongRunTest("AddLoc3",
								new Adder<StorelessUnivariateStatistic>(locked),
								repeatStats),
						new PerfTestUtils.LongRunTest("AddLoc4",
								new Adder<StorelessUnivariateStatistic>(locked),
								repeatStats),
						new PerfTestUtils.LongRunTest("AddLoc5",
								new Adder<StorelessUnivariateStatistic>(locked),
								repeatStats),
						new PerfTestUtils.LongRunTest("AddLoc6",
								new Adder<StorelessUnivariateStatistic>(locked),
								repeatStats),
						new PerfTestUtils.LongRunTest("AddLoc7",
								new Adder<StorelessUnivariateStatistic>(locked),
								repeatStats),
						new PerfTestUtils.LongRunTest("AddLoc8",
								new Adder<StorelessUnivariateStatistic>(locked),
								repeatStats), 								
						new PerfTestUtils.LongRunTest("MeanLocked",
								new Mean<StorelessUnivariateStatistic>(locked),
								repeatStats), 
						new PerfTestUtils.LongRunTest("StdDevLocked",
								new StdDeviation<StorelessUnivariateStatistic>(locked), 
								repeatStats),
						new PerfTestUtils.LongRunTest("QuantileLocked",
								new Quantile<StorelessUnivariateStatistic>(locked), 
								repeatStats)
						);

	}

	@Test
	public void testAddSyn() {
		DescriptiveStatistics synchronous = new SynchronizedDescriptiveStatistics();
		StatisticalSummary[] summary = PerfTestUtils
				.timesAndResultsConcurrently(
						new PerfTestUtils.LongRunTest("AddSyn1", 
								new Adder<UnivariateStatistic>(synchronous),
								repeatStats),
								
						new PerfTestUtils.LongRunTest("AddSyn2",
								new Adder<UnivariateStatistic>(synchronous),
								repeatStats), 
						
						new PerfTestUtils.LongRunTest("MeanSyn",
								new Mean<UnivariateStatistic>(synchronous),
								repeatStats), 
						
						new PerfTestUtils.LongRunTest("StdDevSyn",
								new StdDeviation<UnivariateStatistic>(synchronous), 
								repeatStats),
										
						new PerfTestUtils.LongRunTest("QuantileSyn",
								new Quantile<UnivariateStatistic>(synchronous),
								repeatStats));
	}

	@Test
	public void testAddParallel() throws InterruptedException {
		DescriptiveStatisticalSummary<StorelessUnivariateStatistic> lockFree = 
				new LockfreeDescriptiveStorelessStatistics();
		Publisher<StorelessUnivariateStatistic> pub = 
				new Publisher<StorelessUnivariateStatistic>("pub", repeatStats, lockFree);
		StatisticalSummary[] summary = PerfTestUtils
				.timesAndResultsConcurrently(
						new PerfTestUtils.LongRunTest("AddDis1",
							new Adder<StorelessUnivariateStatistic>(lockFree),
							repeatStats), 
						new PerfTestUtils.LongRunTest("AddDis2",
							new Adder<StorelessUnivariateStatistic>(lockFree),
							repeatStats), 
						new PerfTestUtils.LongRunTest("AddDis3",
							new Adder<StorelessUnivariateStatistic>(lockFree),
							repeatStats), 
						new PerfTestUtils.LongRunTest("AddDis4",
							new Adder<StorelessUnivariateStatistic>(lockFree),
							repeatStats), 
						new PerfTestUtils.LongRunTest("AddDis5",
							new Adder<StorelessUnivariateStatistic>(lockFree),
							repeatStats), 
						new PerfTestUtils.LongRunTest("AddDis6",
							new Adder<StorelessUnivariateStatistic>(lockFree),
							repeatStats),
						new PerfTestUtils.LongRunTest("AddDis7",
							new Adder<StorelessUnivariateStatistic>(lockFree),
							repeatStats), 
						new PerfTestUtils.LongRunTest("AddDis8",
							new Adder<StorelessUnivariateStatistic>(lockFree),
							repeatStats), 
						new PerfTestUtils.LongRunTest("MeanParallel", 
							new Mean<StorelessUnivariateStatistic>(lockFree), 
							repeatStats),
						new PerfTestUtils.LongRunTest("StdDevParallel",
							new StdDeviation<StorelessUnivariateStatistic>(lockFree), 
							repeatStats),
						new PerfTestUtils.LongRunTest("QuantileParallel",
							new Quantile<StorelessUnivariateStatistic>(lockFree), 
							repeatStats)
					);
		pub.stop();
		lockFree.clear();
		lockFree.halt();
	}

	/**
	 * Ａ
	 */
	private static class Publisher<S extends UnivariateStatistic> implements
			Runnable {
		private final String name;
		private volatile boolean stop;
		private volatile boolean delay;
		private volatile long millis = 0L;
		private final Random rand = new Random(Long.MAX_VALUE);
		private final DescriptiveStatisticalSummary<S> stats;
		private final long count;

		public Publisher(String name, long count,
				DescriptiveStatisticalSummary<S> stats) {
			this.count = count;
			this.name = name;
			this.stats = stats;
		}

		public void run() {
			long i = 0;
			while (!stop ) {
				stats.addValue((rand.nextDouble()));
				++i;
				stop = i < count;
				if (delay)
					LockSupport.parkNanos(millis * 1000 * 1000);
			}
		}

		public Publisher<S> stop() {
			stop = true;
			return this;
		}

		public Publisher<S> delay(long millis) {
			delay = true;
			this.millis = millis;
			return this;
		}

		public String getName() {
			return name;
		}
	}
}
