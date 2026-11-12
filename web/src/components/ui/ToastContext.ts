import { createContext } from "react";

export type ToastType = "success" | "error" | "info";

export interface ToastMessage {
  id: string;
  title: string;
  description?: string;
  type?: ToastType;
}

export interface ToastContextType {
  toast: (options: Omit<ToastMessage, "id">) => void;
}

export const ToastContext = createContext<ToastContextType | undefined>(undefined);

