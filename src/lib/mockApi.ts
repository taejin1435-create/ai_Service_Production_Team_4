/**
 * Mock API - 백엔드 없이 프론트엔드 개발을 진행하기 위한 Mock 함수들
 * 
 * 실제 백엔드 API가 준비되면 이 함수들을 실제 API 호출로 대체합니다.
 */

import { User, AnalysisResult, SkinCondition, ZoomArea, SeverityLevel } from '@/types';

// ============ 사용자 인증 Mock API ============

export const mockLogin = async (email: string, password: string): Promise<User> => {
  // 시뮬레이션: 2초 지연
  await new Promise(resolve => setTimeout(resolve, 2000));

  // Mock 검증 (간단한 예시)
  if (!email.includes('@')) {
    throw new Error('유효한 이메일을 입력해주세요.');
  }
  if (password.length < 6) {
    throw new Error('비밀번호는 6자 이상이어야 합니다.');
  }

  // Mock 사용자 반환
  const user: User = {
    id: 'user_' + Math.random().toString(36).substr(2, 9),
    name: email.split('@')[0],
    email,
    createdAt: new Date(),
  };

  // localStorage에 저장
  localStorage.setItem('auth_user', JSON.stringify(user));
  localStorage.setItem('auth_token', 'mock_token_' + Date.now());

  return user;
};

export const mockSignup = async (
  name: string,
  email: string,
  password: string
): Promise<User> => {
  // 시뮬레이션: 2초 지연
  await new Promise(resolve => setTimeout(resolve, 2000));

  // Mock 검증
  if (!name.trim()) {
    throw new Error('이름을 입력해주세요.');
  }
  if (!email.includes('@')) {
    throw new Error('유효한 이메일을 입력해주세요.');
  }
  if (password.length < 6) {
    throw new Error('비밀번호는 6자 이상이어야 합니다.');
  }

  // Mock 사용자 반환
  const user: User = {
    id: 'user_' + Math.random().toString(36).substr(2, 9),
    name,
    email,
    createdAt: new Date(),
  };

  // localStorage에 저장
  localStorage.setItem('auth_user', JSON.stringify(user));
  localStorage.setItem('auth_token', 'mock_token_' + Date.now());

  return user;
};

export const mockLogout = (): void => {
  localStorage.removeItem('auth_user');
  localStorage.removeItem('auth_token');
};

export const mockGetCurrentUser = (): User | null => {
  const userStr = localStorage.getItem('auth_user');
  if (!userStr) return null;
  try {
    return JSON.parse(userStr);
  } catch {
    return null;
  }
};

// ============ 이미지 분석 Mock API ============

// Mock 피부질환 데이터베이스
const mockConditions: Record<string, SkinCondition> = {
  acne: {
    id: 'cond_acne',
    name: '여드름',
    description: '모낭이 피지와 죽은 피부 세포로 막혀서 생기는 염증성 피부질환입니다.',
    severity: 'warning',
    confidence: 85,
    recommendations: [
      '하루 2회 순한 클렌저로 세안하기',
      '비타민 A 함유 제품 사용 고려',
      '피부과 전문의 상담 권장',
    ],
  },
  eczema: {
    id: 'cond_eczema',
    name: '습진',
    description: '피부가 붉어지고 가려운 만성 염증 질환입니다.',
    severity: 'warning',
    confidence: 72,
    recommendations: [
      '보습 크림 정기적 사용',
      '자극적인 물질 피하기',
      '의료 전문가 상담 필수',
    ],
  },
  psoriasis: {
    id: 'cond_psoriasis',
    name: '건선',
    description: '피부 세포가 빠르게 증식하여 두꺼운 비늘 모양의 반점이 생기는 질환입니다.',
    severity: 'danger',
    confidence: 68,
    recommendations: [
      '전문 의료 기관 방문 필수',
      '처방 약물 사용',
      '정기적인 피부 관리',
    ],
  },
  rosacea: {
    id: 'cond_rosacea',
    name: '주사',
    description: '얼굴이 쉽게 붉어지고 혈관이 확장되는 만성 피부질환입니다.',
    severity: 'warning',
    confidence: 76,
    recommendations: [
      '자극적인 제품 피하기',
      '자외선 차단제 필수 사용',
      '피부과 전문의 상담',
    ],
  },
  healthy: {
    id: 'cond_healthy',
    name: '건강한 피부',
    description: '특별한 피부질환이 감지되지 않습니다.',
    severity: 'normal',
    confidence: 95,
    recommendations: [
      '현재의 스킨케어 루틴 유지',
      '충분한 수분 섭취',
      '정기적인 피부 관리 계속',
    ],
  },
};

// Mock 분석 결과 생성 함수
const generateMockAnalysis = (imageUrl: string, userId: string): AnalysisResult => {
  const conditionKeys = Object.keys(mockConditions);
  
  // 랜덤하게 조건 선택 (건강한 피부 확률 40%)
  const randomCondition = Math.random() > 0.4 
    ? conditionKeys[Math.floor(Math.random() * (conditionKeys.length - 1))]
    : 'healthy';

  const overallCondition = mockConditions[randomCondition];

  // Mock 확대 영역 생성 (건강한 피부가 아닌 경우만)
  const zoomAreas: ZoomArea[] = randomCondition !== 'healthy'
    ? [
        {
          id: 'zoom_1',
          x: 20,
          y: 25,
          width: 25,
          height: 25,
          condition: overallCondition,
        },
        {
          id: 'zoom_2',
          x: 55,
          y: 40,
          width: 20,
          height: 20,
          condition: {
            ...overallCondition,
            confidence: overallCondition.confidence - 10,
          },
        },
      ]
    : [];

  return {
    id: 'analysis_' + Math.random().toString(36).substr(2, 9),
    userId,
    imageUrl,
    timestamp: new Date(),
    overallCondition,
    zoomAreas,
    isAnalyzing: false,
  };
};

export const mockAnalyzeImage = async (
  imageUrl: string,
  userId: string
): Promise<AnalysisResult> => {
  // 시뮬레이션: 3-5초 지연 (분석 진행 중)
  const delay = 3000 + Math.random() * 2000;
  await new Promise(resolve => setTimeout(resolve, delay));

  const result = generateMockAnalysis(imageUrl, userId);

  // localStorage에 분석 결과 저장
  const history = JSON.parse(localStorage.getItem('analysis_history') || '[]');
  history.push(result);
  localStorage.setItem('analysis_history', JSON.stringify(history));

  return result;
};

export const mockGetAnalysisHistory = (userId: string): AnalysisResult[] => {
  const history = JSON.parse(localStorage.getItem('analysis_history') || '[]');
  return history.filter((analysis: AnalysisResult) => analysis.userId === userId);
};

// ============ 유틸리티 함수 ============

/**
 * 파일을 Base64 데이터 URL로 변환
 */
export const fileToDataUrl = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
};

/**
 * 이미지 파일 검증
 */
export const validateImageFile = (file: File): { valid: boolean; error?: string } => {
  const validTypes = ['image/jpeg', 'image/png', 'image/webp'];
  const maxSize = 10 * 1024 * 1024; // 10MB

  if (!validTypes.includes(file.type)) {
    return {
      valid: false,
      error: 'JPEG, PNG, WebP 형식의 이미지만 업로드 가능합니다.',
    };
  }

  if (file.size > maxSize) {
    return {
      valid: false,
      error: '파일 크기는 10MB 이하여야 합니다.',
    };
  }

  return { valid: true };
};
