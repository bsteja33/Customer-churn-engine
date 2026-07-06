"use client";

import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";

/** Map of every form field to its current value (or null when unset). */
export type FormValues = Record<string, string | number | null>;

interface FormState {
  values: FormValues;
  updateField: (key: string, value: string | number | null) => void;
  setValues: (values: FormValues) => void;
  reset: () => void;
  loadPreset: (preset: { values: FormValues }) => void;
}

const STORAGE_KEY = "churn.form.v1";
const STORAGE_VERSION = 1;

export const useFormStore = create<FormState>()(
  persist(
    (set) => ({
      values: {},

      updateField: (key, value) =>
        set((state) => ({ values: { ...state.values, [key]: value } })),

      setValues: (values) => set(() => ({ values: { ...values } })),

      reset: () => set(() => ({ values: {} })),

      loadPreset: (preset) =>
        set(() => ({ values: { ...preset.values } })),
    }),
    {
      name: STORAGE_KEY,
      version: STORAGE_VERSION,
      storage: createJSONStorage(() => {
        // SSR guard: rehydration happens on the client only.
        if (typeof window === "undefined") {
          return {
            getItem: () => null,
            setItem: () => undefined,
            removeItem: () => undefined,
          };
        }
        return window.localStorage;
      }),
      partialize: (state) => ({ values: state.values }),
    }
  )
);
