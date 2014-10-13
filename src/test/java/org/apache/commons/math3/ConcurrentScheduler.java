package org.apache.commons.math3;

import java.util.Collection;
import java.util.List;
import java.util.concurrent.AbstractExecutorService;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.FutureTask;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import org.junit.runners.model.RunnerScheduler;

import com.lmax.disruptor.util.DaemonThreadFactory;

public class ConcurrentScheduler implements RunnerScheduler {
	private final int availableThreads = Runtime.getRuntime()
			.availableProcessors();
	private final long nanoSecondsTimeOut;
	private final ExecutorService service;

	ConcurrentScheduler() {
		this(1, Long.MAX_VALUE);
	}

	ConcurrentScheduler(final int threads, final long nanoSecondsTimeOut) {
		service = threads < 1 ? new AbstractExecutorService() {
			
			public void execute(Runnable command) {
				command.run();
			}
			
			public List<Runnable> shutdownNow() {
				return null;
			}
			
			public void shutdown() {
			}
			
			public boolean isTerminated() {
				return true;
			}
			
			public boolean isShutdown() {
				return true;
			}
			
			public boolean awaitTermination(long timeout, TimeUnit unit)
					throws InterruptedException {
				return true;
			}
		} : Executors
				.newFixedThreadPool(Math.min(threads, availableThreads),DaemonThreadFactory.INSTANCE);
		this.nanoSecondsTimeOut = nanoSecondsTimeOut;
	}

	public void schedule(Runnable childStatement) {
		service.submit(childStatement);
	}

	public void finished() {
		try {
			service.shutdown();
			service.awaitTermination(nanoSecondsTimeOut, TimeUnit.NANOSECONDS);
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
			throw new RuntimeException("Got interrupted", e);
		}
	}
}
