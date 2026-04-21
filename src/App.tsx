import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import NotFound from "@/pages/NotFound";
import { Route, Switch } from "wouter";
import ErrorBoundary from "./components/ErrorBoundary";
import { ThemeProvider } from "./contexts/ThemeContext";
import { AuthProvider } from "./contexts/AuthContext";
import { AnalysisProvider } from "./contexts/AnalysisContext";
import LoginPage from "./pages/LoginPage";
import SignupPage from "./pages/SignupPage";
import AnalysisPage from "./pages/AnalysisPage";

/**
 * Router - 경로 설정을 담당하는 컴포넌트
 */
function Router() {
  return (
    <Switch>
      {/* 루트 주소(/)로 접속했을 때 바로 로그인 페이지가 나오도록 함. */}
      <Route path="/" component={LoginPage} />
      
      <Route path="/login" component={LoginPage} />
      <Route path="/signup" component={SignupPage} />
      <Route path="/analysis" component={AnalysisPage} />
      
      {/* 404 페이지 설정 */}
      <Route path="/404" component={NotFound} />
      <Route component={NotFound} />
    </Switch>
  );
}

/**
 * App - 모든 전역 설정(Provider)을 감싸는 최상위 컴포넌트
 */
function App() {
  return (
    <ErrorBoundary>
      {/* defaultTheme="light"를 유지하여 밝은 테마를 기본으로 설정합니다. */}
      <ThemeProvider defaultTheme="light">
        <AuthProvider>
          <AnalysisProvider>
            <TooltipProvider>
              {/* 알림 메시지를 위한 Toaster */}
              <Toaster position="top-center" richColors />
              <Router />
            </TooltipProvider>
          </AnalysisProvider>
        </AuthProvider>
      </ThemeProvider>
    </ErrorBoundary>
  );
}

export default App;

