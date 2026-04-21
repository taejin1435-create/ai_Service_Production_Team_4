/**
 * ImageViewer - 이미지 뷰어 및 돋보기 확대 기능
 * 
 * 업로드된 이미지를 표시하고, 사용자가 특정 부위를 클릭하여 확대할 수 있는 기능을 제공합니다.
 */

import React, { useState, useRef, useEffect } from 'react';
import { ZoomIn, X } from 'lucide-react';
import { ZoomArea } from '@/types';
import { Button } from '@/components/ui/button';

interface ImageViewerProps {
  imageUrl: string;
  zoomAreas?: ZoomArea[];
  onZoomAreaClick?: (area: ZoomArea) => void;
}

interface ZoomState {
  isActive: boolean;
  x: number;
  y: number;
  width: number;
  height: number;
}

export default function ImageViewer({
  imageUrl,
  zoomAreas = [],
  onZoomAreaClick,
}: ImageViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const [zoom, setZoom] = useState<ZoomState>({
    isActive: false,
    x: 0,
    y: 0,
    width: 0,
    height: 0,
  });
  const [imageDimensions, setImageDimensions] = useState({ width: 0, height: 0 });

  // 이미지 로드 시 실제 크기 계산
  useEffect(() => {
    const img = imageRef.current;
    if (img) {
      const handleLoad = () => {
        setImageDimensions({
          width: img.offsetWidth,
          height: img.offsetHeight,
        });
      };
      if (img.complete) {
        handleLoad();
      } else {
        img.addEventListener('load', handleLoad);
        return () => img.removeEventListener('load', handleLoad);
      }
    }
  }, [imageUrl]);

  // 마우스 클릭으로 확대 영역 선택
  const handleImageClick = (e: React.MouseEvent<HTMLImageElement>) => {
    if (!imageRef.current) return;

    const rect = imageRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // 클릭 위치를 백분율로 변환
    const xPercent = (x / rect.width) * 100;
    const yPercent = (y / rect.height) * 100;

    // 확대 영역 크기 (백분율)
    const zoomSize = 30;

    // 확대 영역이 이미지 범위를 벗어나지 않도록 조정
    const startX = Math.max(0, Math.min(xPercent - zoomSize / 2, 100 - zoomSize));
    const startY = Math.max(0, Math.min(yPercent - zoomSize / 2, 100 - zoomSize));

    setZoom({
      isActive: true,
      x: startX,
      y: startY,
      width: zoomSize,
      height: zoomSize,
    });
  };

  const handleCloseZoom = () => {
    setZoom({ ...zoom, isActive: false });
  };

  // 확대 영역 박스 스타일 계산
  const zoomBoxStyle = {
    left: `${zoom.x}%`,
    top: `${zoom.y}%`,
    width: `${zoom.width}%`,
    height: `${zoom.height}%`,
  };

  return (
    <div className="w-full space-y-4">
      {/* 메인 이미지 */}
      <div
        ref={containerRef}
        className="relative bg-gray-100 rounded-lg overflow-hidden border-2 border-gray-200"
      >
        <img
          ref={imageRef}
          src={imageUrl}
          alt="업로드된 이미지"
          onClick={handleImageClick}
          className="w-full h-auto cursor-crosshair"
        />

        {/* 분석 가능한 영역 표시 */}
        {zoomAreas.map((area) => (
          <div
            key={area.id}
            className="absolute border-2 border-orange-400 bg-orange-100 bg-opacity-10 hover:bg-opacity-20 cursor-pointer transition-all"
            style={{
              left: `${area.x}%`,
              top: `${area.y}%`,
              width: `${area.width}%`,
              height: `${area.height}%`,
            }}
            onClick={() => onZoomAreaClick?.(area)}
            title={`클릭: ${area.condition.name}`}
          >
            <div className="absolute top-1 left-1 bg-orange-500 text-white text-xs px-2 py-1 rounded opacity-0 hover:opacity-100 transition-opacity">
              {area.condition.name}
            </div>
          </div>
        ))}

        {/* 확대 박스 */}
        {zoom.isActive && (
          <div
            className="absolute border-4 border-blue-500 bg-blue-100 bg-opacity-20 shadow-lg"
            style={zoomBoxStyle}
          >
            <div className="absolute top-2 right-2 bg-blue-500 text-white rounded-full p-1">
              <ZoomIn className="w-4 h-4" />
            </div>
          </div>
        )}

        {/* 클릭 안내 텍스트 */}
        {!zoom.isActive && zoomAreas.length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-0 hover:bg-opacity-10 transition-all">
            <div className="text-center text-white drop-shadow-lg">
              <ZoomIn className="w-8 h-8 mx-auto mb-2 opacity-0 group-hover:opacity-100" />
              <p className="text-sm font-medium opacity-0 hover:opacity-100">
                이미지를 클릭하여 확대
              </p>
            </div>
          </div>
        )}
      </div>

      {/* 확대 영역 미리보기 */}
      {zoom.isActive && (
        <div className="bg-white border-2 border-blue-200 rounded-lg p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-semibold text-gray-800">확대 영역</h3>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleCloseZoom}
              className="h-8 w-8 p-0"
            >
              <X className="w-4 h-4" />
            </Button>
          </div>

          {/* 확대 미리보기 */}
          <div className="relative bg-gray-100 rounded overflow-hidden border border-gray-300 aspect-square">
            <img
              src={imageUrl}
              alt="확대 미리보기"
              className="w-full h-full object-cover"
              style={{
                objectPosition: `${zoom.x + zoom.width / 2}% ${zoom.y + zoom.height / 2}%`,
                transform: 'scale(3)',
              }}
            />
          </div>

          <p className="text-xs text-gray-500 mt-2">
            위치: ({Math.round(zoom.x)}%, {Math.round(zoom.y)}%)
          </p>
        </div>
      )}

      {/* 안내 텍스트 */}
      <div className="text-sm text-gray-600 bg-blue-50 p-3 rounded-lg border border-blue-200">
        <p>
          💡 <strong>팁:</strong> 이미지의 특정 부위를 클릭하면 해당 영역이 확대됩니다. 
          {zoomAreas.length > 0 && ' 주황색 박스는 분석 가능한 영역입니다.'}
        </p>
      </div>
    </div>
  );
}
