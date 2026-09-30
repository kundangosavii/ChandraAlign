import { useState } from 'react';
import MetricsCard from './MetricsCard';
import ImagePreview from './ImagePreview';
import MatchAnalytics from './MatchAnalytics';
import { Activity, Target, Clock, ChevronLeft, ChevronRight } from 'lucide-react';

export const ResultPanel = ({ results }) => {
  const [matchPage, setMatchPage] = useState(0);

  if (!results) return null;

  const {
    match_image,
    match_image_pages,
    aligned_image,
    rmse,
    inlier_ratio,
    compute_time,
    match_details,
    result_mode,
    aligned_image_status,
  } = results;
  const isSyntheticDemo = result_mode === 'synthetic_demo';
  const isActuallyRegistered = aligned_image_status !== 'preprocessed_source_not_registered';
  const matchPages = Array.isArray(match_image_pages) && match_image_pages.length
    ? match_image_pages
    : [match_image];
  const currentMatchPage = Math.min(matchPage, matchPages.length - 1);
  const totalMatchCount = Number(match_details?.num_input_matches ?? 0);

  return (
    <div className="space-y-6 animate-fadeIn">

      {/* Metrics Row */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <MetricsCard
          title={isSyntheticDemo ? 'RMSE' : 'Root Mean Square Error'}
          value={typeof rmse === 'number' ? rmse.toFixed(4) : rmse}
          unit="px"
          icon={Activity}
          description={isSyntheticDemo ? 'Root Mean Square Error' : 'Geometric misalignment error magnitude'}
        />
        <MetricsCard
          title={isSyntheticDemo ? 'Inlier Ratio' : 'Inlier Matching Ratio'}
          value={typeof inlier_ratio === 'number' ? `${(inlier_ratio * 100).toFixed(1)}%` : inlier_ratio}
          icon={Target}
          description={isSyntheticDemo ? 'Inlier Matching Ratio' : 'Robust feature point consensus ratio'}
        />
        <MetricsCard
          title="Processing Duration"
          value={typeof compute_time === 'number' ? compute_time.toFixed(3) : compute_time}
          unit="sec"
          icon={Clock}
          description="Server compute pipeline latency"
        />
      </div>

      {/* Visual Alignment Outputs */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="flex min-w-0 flex-col gap-2">
          <ImagePreview
            title="Feature Keypoint Matching Map"
            base64String={matchPages[currentMatchPage]}
            badge={isSyntheticDemo ? 'Correspondences' : 'Correspondence'}
            filename={`luna_match_features_${currentMatchPage + 1}.png`}
          />
          <div className="flex items-center justify-between rounded-lg border border-slate-800 bg-slate-900 px-3 py-2 text-xs text-slate-400">
            <button
              type="button"
              className="flex items-center gap-1 rounded px-2 py-1 hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-40"
              disabled={currentMatchPage === 0}
              onClick={() => setMatchPage((page) => Math.max(0, page - 1))}
            >
              <ChevronLeft className="h-4 w-4" />
              Previous 100
            </button>
            <span>
              Matches {totalMatchCount ? currentMatchPage * 100 + 1 : 0}–
              {Math.min((currentMatchPage + 1) * 100, totalMatchCount)}
              {' '}of {totalMatchCount}
            </span>
            <button
              type="button"
              className="flex items-center gap-1 rounded px-2 py-1 hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-40"
              disabled={currentMatchPage >= matchPages.length - 1}
              onClick={() => setMatchPage((page) => Math.min(matchPages.length - 1, page + 1))}
            >
              Next 100
              <ChevronRight className="h-4 w-4" />
            </button>
          </div>
        </div>
        <ImagePreview
          title={
            isSyntheticDemo
              ? 'Final Aligned Surface Output'
              : isActuallyRegistered
                ? 'Final Aligned Surface Output'
                : 'Preprocessed Source (Not Registered)'
          }
          base64String={aligned_image}
          badge={
            isSyntheticDemo
              ? 'Registered'
              : isActuallyRegistered
                ? 'Registered'
                : 'Not registered'
          }
          filename="luna_aligned_surface.png"
        />
      </div>

      <MatchAnalytics details={match_details} />
    </div>
  );
};

export default ResultPanel;