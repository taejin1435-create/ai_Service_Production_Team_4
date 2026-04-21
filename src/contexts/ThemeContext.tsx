import React, { createContext, useContext } from "react"

interface ThemeProviderProps {
  children: React.ReactNode
  defaultTheme?: string
  storageKey?: string
}

export const ThemeProvider = ({
  children,
  defaultTheme = "light",
  storageKey = "vite-ui-theme",
  ...props
}: ThemeProviderProps) => {
  // 현재는 기능을 비워두어 에러만 방지하고, 나중에 다크모드 기능을 넣을 때 확장 가능합니다.
  return <>{children}</>
}

// 에러 방지를 위한 빈 컨텍스트
export const useTheme = () => {
  return { theme: "light", setTheme: () => {} }
}