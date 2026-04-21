/**
 * Skin AI Diagnosis System - Type Definitions
 * 
 * 전체 애플리케이션에서 사용되는 타입을 정의합니다.
 */

// ============ 사용자 관련 타입 ============
export interface User {
    id: string;
    name: string;
    email: string;
    createdAt: Date;
  }
  
  export interface AuthState {
    user: User | null;
    isAuthenticated: boolean;
    isLoading: boolean;
    error: string | null;
  }
  
  // ============ 분석 관련 타입 ============
  export type SeverityLevel = 'normal' | 'warning' | 'danger';
  
  export interface SkinCondition {
    id: string;
    name: string;
    description: string;
    severity: SeverityLevel;
    confidence: number; // 0-100
    recommendations: string[];
  }
  
  export interface ZoomArea {
    id: string;
    x: number; // 좌측 상단 X 좌표 (%)
    y: number; // 좌측 상단 Y 좌표 (%)
    width: number; // 너비 (%)
    height: number; // 높이 (%)
    condition: SkinCondition;
  }
  
  export interface AnalysisResult {
    id: string;
    userId: string;
    imageUrl: string;
    timestamp: Date;
    overallCondition: SkinCondition;
    zoomAreas: ZoomArea[];
    isAnalyzing: boolean;
  }
  
  export interface AnalysisState {
    currentAnalysis: AnalysisResult | null;
    analysisHistory: AnalysisResult[];
    isLoading: boolean;
    error: string | null;
  }
  
  // ============ UI 상태 타입 ============
  export interface ImageUploadState {
    file: File | null;
    preview: string | null;
    isUploading: boolean;
    error: string | null;
  }
  
  export interface ZoomState {
    isZooming: boolean;
    selectedAreaId: string | null;
    zoomLevel: number; // 1-3
  }
  
  // ============ Mock API 응답 타입 ============
  export interface LoginRequest {
    email: string;
    password: string;
  }
  
  export interface SignupRequest {
    name: string;
    email: string;
    password: string;
  }
  
  export interface AnalysisRequest {
    imageUrl: string;
    userId: string;
  }
  
  export interface AnalysisResponse {
    success: boolean;
    data?: AnalysisResult;
    error?: string;
  }
  