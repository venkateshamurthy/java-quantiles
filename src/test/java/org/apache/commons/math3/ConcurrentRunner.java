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

package org.apache.commons.math3;

import org.junit.runners.BlockJUnit4ClassRunner;
import org.junit.runners.model.InitializationError;

/**
 * A test runner that runs on concurrently
 */
public class ConcurrentRunner extends BlockJUnit4ClassRunner {

	private final Concurrency conc;

	/**
	 * Simple constructor.
	 * 
	 * @param testClass
	 *            Class to test.
	 * @throws InitializationError
	 *             if default runner cannot be built.
	 */
	public ConcurrentRunner(final Class<?> testClass)
			throws InitializationError {
		super(testClass);
		conc = testClass.getAnnotation(Concurrency.class);
		setScheduler(new ConcurrentScheduler(conc.threads(), conc.timeOutNanos()));
	}
}
