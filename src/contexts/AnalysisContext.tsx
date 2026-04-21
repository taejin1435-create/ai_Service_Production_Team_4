/**
 * AnalysisContext - 이미지 분석 결과 상태 관리
 * 
 * 분석 결과, 분석 히스토리 등의 상태를 관리합니다.
 */

import React, { createContext, useContext, useState } from 'react';
import { AnalysisResult, AnalysisState } from '@/types';
import { mockAnalyzeImage, mockGetAnalysisHistory } from '@/lib/mockApi';

interface AnalysisContextType extends AnalysisState {
  analyzeImage: (imageUrl: string, userId: string) => Promise<void>;
  loadHistory: (userId: string) => void;
  clearCurrent: () => void;
}

const AnalysisContext = createContext<AnalysisContextType | undefined>(undefined);

export const AnalysisProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [currentAnalysis, setCurrentAnalysis] = useState<AnalysisResult | null>(null);
  const [analysisHistory, setAnalysisHistory] = useState<AnalysisResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const analyzeImage = async (imageUrl: string, userId: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await mockAnalyzeImage(imageUrl, userId);
      setCurrentAnalysis(result);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : '분석 실패';
      setError(errorMessage);
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  const loadHistory = (userId: string) => {
    const history = mockGetAnalysisHistory(userId);
    setAnalysisHistory(history);
  };

  const clearCurrent = () => {
    setCurrentAnalysis(null);
    setError(null);
  };

  const value: AnalysisContextType = {
    currentAnalysis,
    analysisHistory,
    isLoading,
    error,
    analyzeImage,
    loadHistory,
    clearCurrent,
  };

  return (
    <AnalysisContext.Provider value={value}>{children}</AnalysisContext.Provider>
  );
};

export const useAnalysis = (): AnalysisContextType => {
  const context = useContext(AnalysisContext);
  if (!context) {
    throw new Error('useAnalysis must be used within an AnalysisProvider');
  }
  return context;
};
