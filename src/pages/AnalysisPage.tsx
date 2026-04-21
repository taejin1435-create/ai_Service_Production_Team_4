/**
 * AnalysisPage - 메인 분석 페이지 (드래그 가능한 AI 감지 박스 통합)
 */

import React, { useState, useRef, useEffect } from 'react';
import { useLocation } from 'wouter';
import { useAuth } from '@/contexts/AuthContext';
import { useAnalysis } from '@/contexts/AnalysisContext';
import ImageUploader from '@/components/ImageUploader';
import ImageViewer from '@/components/ImageViewer';
import AnalysisResult from '@/components/AnalysisResult';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { LogOut, Loader2, RotateCcw, Crosshair, AlertCircle, Move } from 'lucide-react';
import { fileToDataUrl } from '@/lib/mockApi';

export default function AnalysisPage() {
  const [, setLocation] = useLocation();
  const { user, logout } = useAuth();
  const { currentAnalysis, isLoading, analyzeImage, clearCurrent } = useAnalysis();
  const [uploadedImage, setUploadedImage] = useState<{ file: File; preview: string } | null>(null);
  const [selectedZoomArea, setSelectedZoomArea] = useState<string | null>(null);

  // 1. 상태 관리(State): 고정되었던 데이터를 수정 가능한 상태로 변경
  const [detections, setDetections] = useState([
    { id: 'det-1', label: '여드름', x: 42, y: 32, width: 8, height: 8 },
    { id: 'det-2', label: '피지', x: 55, y: 48, width: 6, height: 6 },
    { id: 'det-3', label: '아토피', x: 30, y: 60, width: 12, height: 10 },
  ]);

  // 드래그를 위한 참조(Ref) 및 상태
  const [draggingId, setDraggingId] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const offsetRef = useRef({ x: 0, y: 0 });

  // 로그인 확인
  if (!user) {
    setLocation('/login');
    return null;
  }

  // 2. 드래그 로직: 마우스 이벤트 핸들러
  const handleMouseDown = (e: React.MouseEvent, id: string, currentX: number, currentY: number) => {
    e.preventDefault();
    e.stopPropagation();
    setDraggingId(id);
    setSelectedZoomArea(id);

    const container = containerRef.current;
    if (!container) return;

    const rect = container.getBoundingClientRect();
    // 마우스 클릭 지점과 박스 왼쪽 상단 사이의 간격(%) 계산
    const mouseXPercent = ((e.clientX - rect.left) / rect.width) * 100;
    const mouseYPercent = ((e.clientY - rect.top) / rect.height) * 100;

    offsetRef.current = {
      x: mouseXPercent - currentX,
      y: mouseYPercent - currentY,
    };
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!draggingId || !containerRef.current) return;

    const rect = containerRef.current.getBoundingClientRect();
    // 새로운 위치 계산
    let newX = ((e.clientX - rect.left) / rect.width) * 100 - offsetRef.current.x;
    let newY = ((e.clientY - rect.top) / rect.height) * 100 - offsetRef.current.y;

    // 이미지 경계를 벗어나지 않도록 제한 (Clamping)
    newX = Math.max(0, Math.min(newX, 100 - (detections.find(d => d.id === draggingId)?.width || 0)));
    newY = Math.max(0, Math.min(newY, 100 - (detections.find(d => d.id === draggingId)?.height || 0)));

    setDetections((prev) =>
      prev.map((det) => (det.id === draggingId ? { ...det, x: newX, y: newY } : det))
    );
  };

  const handleMouseUp = () => {
    setDraggingId(null);
  };

  const handleImageSelected = async (file: File, preview: string) => {
    setUploadedImage({ file, preview });
    clearCurrent();
    setSelectedZoomArea(null);

    const dataUrl = await fileToDataUrl(file);
    await analyzeImage(dataUrl, user.id);
  };

  const handleLogout = () => {
    logout();
    setLocation('/login');
  };

  const handleReset = () => {
    setUploadedImage(null);
    clearCurrent();
    setSelectedZoomArea(null);
  };

  return (
    <div 
      className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50"
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp} // 마우스가 화면 밖으로 나가도 드래그 중지
    >
      {/* 헤더 */}
      <header className="bg-white border-b border-gray-200 shadow-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-blue-600 flex items-center gap-2">
              <Crosshair className="w-6 h-6" /> Skin AI Diagnosis
            </h1>
            <p className="text-sm text-gray-600 font-medium">피부질환 AI 진단 시스템</p>
          </div>

          <div className="flex items-center gap-4">
            <div className="text-right hidden sm:block">
              <p className="text-sm font-semibold text-gray-800">{user.name}님</p>
              <p className="text-xs text-gray-500">{user.email}</p>
            </div>
            <Button
              onClick={handleLogout}
              variant="outline"
              size="sm"
              className="flex items-center gap-2 border-gray-300 hover:bg-red-50 hover:text-red-600 hover:border-red-200 transition-colors"
            >
              <LogOut className="w-4 h-4" />
              로그아웃
            </Button>
          </div>
        </div>
      </header>

      {/* 메인 콘텐츠 */}
      <main className="container mx-auto px-4 py-8">
        {!uploadedImage ? (
          <div className="max-w-2xl mx-auto mt-10">
            <Card className="shadow-2xl border-0 ring-1 ring-gray-200">
              <CardHeader className="pb-2">
                <CardTitle className="text-center text-xl text-gray-800">분석할 얼굴 사진을 업로드하세요</CardTitle>
              </CardHeader>
              <CardContent>
                <ImageUploader onImageSelected={handleImageSelected} isLoading={isLoading} />
              </CardContent>
            </Card>
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 animate-in fade-in duration-500">
            {/* 좌측: 이미지 뷰어 + 드래그 레이어 */}
            <div className="space-y-4">
              <Card className="shadow-xl border-0 overflow-hidden">
                <CardHeader className="bg-white border-b flex flex-row items-center justify-between py-4">
                  <CardTitle className="text-lg font-bold text-gray-700 flex items-center gap-2">
                    <AlertCircle className="w-5 h-5 text-blue-500" /> 분석 이미지 및 영역 조정
                  </CardTitle>
                  <Button
                    onClick={handleReset}
                    variant="ghost"
                    size="sm"
                    className="flex items-center gap-2 text-gray-500 hover:text-blue-600"
                  >
                    <RotateCcw className="w-4 h-4" />
                    새 이미지
                  </Button>
                </CardHeader>
                
                {/* 3. 컨테이너 참조: 이미지의 실제 픽셀 크기를 알아내기 위함 */}
                <CardContent className="p-0 relative bg-slate-900 select-none">
                  {isLoading ? (
                    <div className="flex flex-col items-center justify-center py-32 space-y-4 bg-white/90">
                      <Loader2 className="w-12 h-12 text-blue-500 animate-spin" />
                      <p className="text-gray-600 font-bold">AI가 피부를 정밀 분석 중입니다...</p>
                    </div>
                  ) : (
                    <div className="relative" ref={containerRef}>
                      <ImageViewer
                        imageUrl={uploadedImage.preview}
                        zoomAreas={currentAnalysis?.zoomAreas}
                        onZoomAreaClick={(area) => setSelectedZoomArea(area.id)}
                      />
                      
                      {/* 드래그 가능한 감지 박스 오버레이 */}
                      {detections.map((det) => (
                        <div
                          key={det.id}
                          onMouseDown={(e) => handleMouseDown(e, det.id, det.x, det.y)}
                          className={`absolute border-2 rounded-sm transition-all duration-200 ${
                            draggingId === det.id 
                            ? 'border-yellow-400 bg-yellow-400/30 cursor-grabbing z-30 scale-105 shadow-2xl' 
                            : selectedZoomArea === det.id
                            ? 'border-yellow-400 bg-yellow-400/10 cursor-grab z-20 shadow-lg'
                            : 'border-red-500 bg-red-500/10 hover:bg-red-500/30 cursor-grab z-10'
                          }`}
                          style={{
                            left: `${det.x}%`,
                            top: `${det.y}%`,
                            width: `${det.width}%`,
                            height: `${det.height}%`,
                            touchAction: 'none' // 모바일 스크롤 방지
                          }}
                        >
                          <span className="absolute -top-7 left-0 bg-red-600 text-white text-[11px] px-2 py-0.5 rounded shadow-lg font-bold whitespace-nowrap flex items-center gap-1">
                            <Move className="w-3 h-3" />
                            {det.label}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                  
                  <div className="p-4 bg-blue-600 text-white text-sm flex items-center justify-center gap-2 font-medium">
                    <Move className="w-4 h-4 animate-pulse" />
                    <span>진단 영역이 정확하지 않나요? 박스를 드래그하여 직접 옮길 수 있습니다.</span>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* 우측: 분석 결과 */}
            <div className="animate-in slide-in-from-right duration-700">
              {currentAnalysis && !isLoading ? (
                <AnalysisResult
                  result={currentAnalysis}
                  onZoomAreaClick={(areaId) => setSelectedZoomArea(areaId)}
                />
              ) : isLoading ? (
                <Card className="shadow-lg border-0 h-full flex items-center justify-center min-h-[400px]">
                  <CardContent>
                    <div className="text-center space-y-4">
                      <Loader2 className="w-12 h-12 text-blue-200 animate-spin mx-auto" />
                      <p className="text-gray-400 font-medium">진단 보고서 생성 중...</p>
                    </div>
                  </CardContent>
                </Card>
              ) : null}
            </div>
          </div>
        )}

        {/* 정보 섹션 */}
        <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-6">
          <Card className="border-0 shadow-md bg-white/60 backdrop-blur-sm">
            <CardHeader><CardTitle className="text-lg">📋 사용 방법</CardTitle></CardHeader>
            <CardContent className="text-sm text-gray-600 space-y-2">
              <p>1. 사진을 업로드하고 분석을 기다립니다.</p>
              <p>2. 생성된 박스를 마우스로 끌어 위치를 조정하세요.</p>
              <p>3. 박스를 클릭하여 상세 분석 결과를 확인합니다.</p>
            </CardContent>
          </Card>

          <Card className="border-0 shadow-md bg-white/60 backdrop-blur-sm">
            <CardHeader><CardTitle className="text-lg text-amber-600">⚠️ 주의사항</CardTitle></CardHeader>
            <CardContent className="text-sm text-gray-600 space-y-2">
              <p>• AI 분석은 참고용 지표입니다.</p>
              <p>• 정확한 진단은 전문의와 상담하세요.</p>
            </CardContent>
          </Card>

          <Card className="border-0 shadow-md bg-white/60 backdrop-blur-sm">
            <CardHeader><CardTitle className="text-lg text-blue-600">💡 팁</CardTitle></CardHeader>
            <CardContent className="text-sm text-gray-600 space-y-2">
              <p>• 드래그 기능을 통해 더 정확한 부위를 지정하세요.</p>
              <p>• 고해상도 사진일수록 AI 정확도가 올라갑니다.</p>
            </CardContent>
          </Card>
        </div>
      </main>

      {/* 푸터 */}
      <footer className="bg-white/80 border-t border-gray-200 mt-20 backdrop-blur-md">
        <div className="container mx-auto px-4 py-8 text-center text-sm text-gray-500">
          <p className="font-semibold text-gray-700 mb-2">Skin AI Diagnosis System</p>
          <p>© 2026 Skin AI Lab. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}