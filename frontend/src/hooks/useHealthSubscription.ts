"use client";

import { useEffect } from "react";
import { subscribeHealth } from "../store/useHealthStore";

/**
 * Subscribe the current component to the shared health-polling loop.
 * Internally manages the subscribe/unsubscribe lifecycle so consumers
 * only need to call this hook once with no dependencies.
 */
export function useHealthSubscription(): void {
  useEffect(() => subscribeHealth(), []);
}
