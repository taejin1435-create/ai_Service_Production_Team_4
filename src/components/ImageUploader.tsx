import React, { useRef, useState } from 'react';
import { Upload, AlertCircle } from 'lucide-react';
import { validateImageFile } from '@/lib/mockApi';
import { Alert, AlertDescription } from '@/components/ui/alert';

interface ImageUploaderProps {
  onImageSelected: (file: File, preview: string) => void;
  isLoading?: boolean;
}

export default function ImageUploader({ onImageSelected, isLoading = false }: ImageUploaderProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFile = async (file: File) => {
    setError(null);

    // 파일 검증
    const validation = validateImageFile(file);
    if (!validation.valid) {
      setError(validation.error || '유효하지 않은 파일입니다.');
      return;
    }

    // 이미지 미리보기 생성
    const reader = new FileReader();
    reader.onload = (e) => {
      const preview = e.target?.result as string;
      onImageSelected(file, preview);
    };
    reader.onerror = () => {
      setError('파일을 읽을 수 없습니다.');
    };
    reader.readAsDataURL(file);
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleFile(files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.currentTarget.files;
    if (files && files.length > 0) {
      handleFile(files[0]);
    }
  };

  const dragClass = dragActive
    ? 'border-blue-500 bg-blue-50'
    : 'border-gray-300 bg-gray-50 hover:border-blue-400 hover:bg-blue-50';

  return (
    <div className="w-full">
      <div
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`relative border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${dragClass} ${
          isLoading ? 'opacity-50 cursor-not-allowed' : ''
        }`}
        style={{ pointerEvents: isLoading ? 'none' : 'auto' }}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleChange}
          disabled={isLoading}
          className="hidden"
        />

        <div className="flex flex-col items-center justify-center space-y-3">
          <div className="text-4xl text-blue-500">
            <Upload className="w-12 h-12 mx-auto" />
          </div>
          <div>
            <p className="text-lg font-semibold text-gray-800">이미지를 업로드하세요</p>
            <p className="text-sm text-gray-600 mt-1">
              얼굴 사진을 드래그 앤 드롭하거나 클릭하여 선택
            </p>
          </div>
          <p className="text-xs text-gray-500">
            지원 형식: JPEG, PNG, WebP (최대 10MB)
          </p>
        </div>
      </div>

      {error && (
        <Alert variant="destructive" className="mt-4">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
    </div>
  );
}