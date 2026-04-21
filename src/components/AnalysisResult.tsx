/**
 * AnalysisResult - 분석 결과 표시 컴포넌트
 * 
 * 피부질환 분석 결과를 시각적으로 표시합니다.
 */

import React from 'react';
import { AnalysisResult as AnalysisResultType, SeverityLevel } from '@/types';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { AlertCircle, CheckCircle2, AlertTriangle, AlertOctagon } from 'lucide-react';
import { Badge } from '@/components/ui/badge';

interface AnalysisResultProps {
  result: AnalysisResultType;
  onZoomAreaClick?: (areaId: string) => void;
}

const getSeverityColor = (severity: SeverityLevel) => {
  switch (severity) {
    case 'normal':
      return 'bg-green-50 border-green-200 text-green-900';
    case 'warning':
      return 'bg-yellow-50 border-yellow-200 text-yellow-900';
    case 'danger':
      return 'bg-red-50 border-red-200 text-red-900';
    default:
      return 'bg-gray-50 border-gray-200 text-gray-900';
  }
};

const getSeverityIcon = (severity: SeverityLevel) => {
  switch (severity) {
    case 'normal':
      return <CheckCircle2 className="w-5 h-5 text-green-600" />;
    case 'warning':
      return <AlertTriangle className="w-5 h-5 text-yellow-600" />;
    case 'danger':
      return <AlertOctagon className="w-5 h-5 text-red-600" />;
    default:
      return <AlertCircle className="w-5 h-5 text-gray-600" />;
  }
};

const getSeverityLabel = (severity: SeverityLevel) => {
  switch (severity) {
    case 'normal':
      return '정상';
    case 'warning':
      return '주의 필요';
    case 'danger':
      return '긴급';
    default:
      return '미분류';
  }
};

export default function AnalysisResult({
  result,
  onZoomAreaClick,
}: AnalysisResultProps) {
  const { overallCondition, zoomAreas } = result;

  return (
    <div className="space-y-4">
      {/* 전체 진단 결과 */}
      <Card className={`border-2 ${getSeverityColor(overallCondition.severity)}`}>
        <CardHeader>
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-3">
              {getSeverityIcon(overallCondition.severity)}
              <div>
                <CardTitle className="text-xl">{overallCondition.name}</CardTitle>
                <CardDescription className="mt-1">
                  {overallCondition.description}
                </CardDescription>
              </div>
            </div>
            <Badge
              variant={
                overallCondition.severity === 'normal'
                  ? 'default'
                  : overallCondition.severity === 'warning'
                    ? 'secondary'
                    : 'destructive'
              }
            >
              {getSeverityLabel(overallCondition.severity)}
            </Badge>
          </div>
        </CardHeader>

        <CardContent className="space-y-4">
          {/* 신뢰도 */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-700">분석 신뢰도</span>
              <span className="text-sm font-semibold text-gray-900">
                {overallCondition.confidence}%
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className={`h-2 rounded-full transition-all ${
                  overallCondition.severity === 'normal'
                    ? 'bg-green-500'
                    : overallCondition.severity === 'warning'
                      ? 'bg-yellow-500'
                      : 'bg-red-500'
                }`}
                style={{ width: `${overallCondition.confidence}%` }}
              />
            </div>
          </div>

          {/* 권장사항 */}
          <div>
            <h4 className="font-semibold text-gray-800 mb-2">권장사항</h4>
            <ul className="space-y-2">
              {overallCondition.recommendations.map((rec, idx) => (
                <li key={idx} className="flex items-start gap-2 text-sm text-gray-700">
                  <span className="text-blue-500 font-bold mt-0.5">•</span>
                  <span>{rec}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* 분석 시간 */}
          <div className="text-xs text-gray-500 pt-2 border-t">
            분석 시간: {new Date(result.timestamp).toLocaleString('ko-KR')}
          </div>
        </CardContent>
      </Card>

      {/* 세부 분석 영역 */}
      {zoomAreas.length > 0 && (
        <div>
          <h3 className="font-semibold text-gray-800 mb-3">세부 분석 영역</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {zoomAreas.map((area) => (
              <Card
                key={area.id}
                className={`cursor-pointer transition-all hover:shadow-lg border-2 ${getSeverityColor(
                  area.condition.severity
                )}`}
                onClick={() => onZoomAreaClick?.(area.id)}
              >
                <CardContent className="pt-4">
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center gap-2">
                      {getSeverityIcon(area.condition.severity)}
                      <div>
                        <p className="font-semibold text-sm">{area.condition.name}</p>
                        <p className="text-xs text-gray-600">
                          위치: ({area.x.toFixed(0)}%, {area.y.toFixed(0)}%)
                        </p>
                      </div>
                    </div>
                    <Badge variant="outline" className="text-xs">
                      {area.condition.confidence}%
                    </Badge>
                  </div>

                  <div className="w-full bg-gray-200 rounded-full h-1.5">
                    <div
                      className={`h-1.5 rounded-full ${
                        area.condition.severity === 'normal'
                          ? 'bg-green-500'
                          : area.condition.severity === 'warning'
                            ? 'bg-yellow-500'
                            : 'bg-red-500'
                      }`}
                      style={{ width: `${area.condition.confidence}%` }}
                    />
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      )}

      {/* 의료 전문가 상담 권장 */}
      {overallCondition.severity !== 'normal' && (
        <div className="bg-blue-50 border-2 border-blue-200 rounded-lg p-4">
          <div className="flex gap-3">
            <AlertCircle className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
            <div>
              <p className="font-semibold text-blue-900 text-sm">의료 전문가 상담 권장</p>
              <p className="text-sm text-blue-800 mt-1">
                정확한 진단과 치료를 위해 피부과 전문의의 상담을 받으시기 바랍니다.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
