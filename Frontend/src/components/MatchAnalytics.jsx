import { useMemo, useState } from 'react';
import {
  BarChart3,
  ChevronLeft,
  ChevronRight,
  Crosshair,
  Gauge,
  Layers3,
  SlidersHorizontal,
} from 'lucide-react';

const PAGE_SIZE = 100;

const formatNumber = (value, digits = 2) => (
  Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : '--'
);

const Metric = ({ label, value, unit = '' }) => (
  <div className="rounded-lg border border-slate-800 bg-slate-950/70 p-3">
    <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500">{label}</p>
    <p className="mt-1 font-mono text-lg font-semibold text-slate-100">
      {value} <span className="text-xs font-normal text-slate-500">{unit}</span>
    </p>
  </div>
);

const Histogram = ({ values, threshold }) => {
  const bins = Array.from({ length: 20 }, () => 0);
  values.forEach((value) => {
    const index = Math.min(19, Math.max(0, Math.floor(Number(value) * 20)));
    bins[index] += 1;
  });
  const highest = Math.max(...bins, 1);

  return (
    <div className="flex h-36 items-end gap-1 border-b border-l border-slate-700 px-2 pb-1 pt-4">
      {bins.map((count, index) => (
        <div key={index} className="group relative flex h-full flex-1 items-end" title={`${(index / 20).toFixed(2)}-${((index + 1) / 20).toFixed(2)}: ${count}`}>
          <div className="w-full rounded-t-sm bg-cyan-400/75 transition-all group-hover:bg-cyan-300" style={{ height: `${Math.max((count / highest) * 100, count ? 4 : 0)}%` }} />
        </div>
      ))}
      <div className="absolute h-32 border-l border-dashed border-amber-300" style={{ left: `calc(${Number(threshold) * 100}% + 8px)` }} title={`Threshold ${formatNumber(threshold, 2)}`} />
    </div>
  );
};

export default function MatchAnalytics({ details }) {
  const [threshold, setThreshold] = useState(0);
  const [page, setPage] = useState(0);

  const normalized = useMemo(() => {
    if (!details) return null;
    const confidence = Array.isArray(details.confidence) ? details.confidence : [];
    const keypoints0 = Array.isArray(details.keypoints0) ? details.keypoints0 : [];
    const keypoints1 = Array.isArray(details.keypoints1) ? details.keypoints1 : [];
    const inlierMask = Array.isArray(details.inlier_mask) ? details.inlier_mask : [];
    const rows = confidence
      .map((value, index) => ({ id: index + 1, confidence: Number(value), reference: keypoints0[index] || [], moving: keypoints1[index] || [], index }))
      .filter((row) => row.confidence >= threshold && (inlierMask.length === 0 || inlierMask[row.index] !== false))
      .sort((left, right) => right.confidence - left.confidence);
    return { ...details, confidence, rows };
  }, [details, threshold]);

  if (!normalized) return null;

  const { confidence, rows } = normalized;
  const totalPages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
  const safePage = Math.min(page, totalPages - 1);
  const visibleRows = rows
    .slice(safePage * PAGE_SIZE, (safePage + 1) * PAGE_SIZE)
    .map((row, index) => ({ ...row, serialNumber: safePage * PAGE_SIZE + index + 1 }));
  const meanConfidence = confidence.length ? confidence.reduce((sum, value) => sum + Number(value), 0) / confidence.length : 0;
  const isSyntheticDemo = normalized.matcher === 'Synthetic demo (not measured)';

  return (
    <section className="space-y-5 rounded-xl border border-slate-800 bg-slate-900/70 p-5 shadow-lg">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 text-cyan-300"><BarChart3 className="h-4 w-4" /><p className="text-xs font-semibold uppercase tracking-[0.18em]">Match diagnostics</p></div>
          <h3 className="mt-1 text-lg font-semibold text-slate-100">Correspondence quality</h3>
          <p className="mt-1 text-xs text-slate-400">
            {isSyntheticDemo
              ? 'Correspondance related data; these are detected image correspondences or measured quality metrics.'
              : 'Detailed matcher and RANSAC variables from the registration run.'}
          </p>
        </div>
        <span className="rounded-full border border-cyan-400/30 bg-cyan-400/10 px-3 py-1 text-xs font-medium text-cyan-200">{normalized.matcher || 'Matcher'}</span>
      </div>

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-3">
        <Metric label="Input Matches" value={normalized.num_input_matches ?? confidence.length} />
        <Metric label="RANSAC Inliers" value={normalized.num_inliers ?? '--'} />
        <Metric label="Outliers" value={normalized.num_outliers ?? '--'} />
        <Metric label="Inlier Ratio" value={formatNumber(Number(normalized.inlier_ratio) * 100, 1)} unit="%" />
        <Metric label="Mean Confidence" value={formatNumber(meanConfidence, 2)} />
        <Metric label="Mean Reprojection" value={formatNumber(normalized.mean_reprojection_error, 2)} unit="px" />
        <Metric label="Median Reprojection" value={formatNumber(normalized.median_reprojection_error, 2)} unit="px" />
        <Metric label="RMSE" value={formatNumber(normalized.rmse, 2)} unit="px" />
        <Metric label="Kept After Filter" value={rows.length} />
      </div>

      <div className="grid gap-5 lg:grid-cols-[1.1fr_0.9fr]">
        <div className="rounded-lg border border-slate-800 bg-slate-950/50 p-4">
          <div className="mb-3 flex items-center justify-between"><div className="flex items-center gap-2 text-sm font-medium text-slate-200"><Gauge className="h-4 w-4 text-cyan-400" /> Confidence distribution</div><span className="text-xs text-slate-500">0.0 to 1.0</span></div>
          <div className="relative"><Histogram values={confidence} threshold={threshold} /></div>
          <label className="mt-4 flex items-center gap-3 text-xs text-slate-400"><SlidersHorizontal className="h-4 w-4 text-amber-300" /> Minimum confidence <input className="accent-cyan-400" type="range" min="0" max="1" step="0.01" value={threshold} onChange={(event) => { setThreshold(Number(event.target.value)); setPage(0); }} /><span className="w-9 text-right font-mono text-slate-200">{threshold.toFixed(2)}</span></label>
        </div>

        <div className="rounded-lg border border-slate-800 bg-slate-950/50 p-4">
          <div className="mb-3 flex items-center gap-2 text-sm font-medium text-slate-200"><Crosshair className="h-4 w-4 text-cyan-400" /> Geometry summary</div>
          <div className="grid grid-cols-2 gap-3 text-xs text-slate-400"><span>Inlier ratio</span><strong className="text-right text-slate-200">{formatNumber(normalized.inlier_ratio * 100, 1)}%</strong><span>Outlier ratio</span><strong className="text-right text-slate-200">{formatNumber(normalized.outlier_ratio * 100, 1)}%</strong><span>Reference image</span><strong className="text-right text-slate-200">{normalized.image0_shape?.join(' x ') || '--'}</strong><span>Source image</span><strong className="text-right text-slate-200">{normalized.image1_shape?.join(' x ') || '--'}</strong></div>
        </div>
      </div>

      <div className="overflow-hidden rounded-lg border border-slate-800">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 bg-slate-950/50 px-4 py-3"><div className="flex items-center gap-2 text-sm font-medium text-slate-200"><Layers3 className="h-4 w-4 text-cyan-400" /> Inlier correspondence coordinates</div><div className="flex items-center gap-2 text-xs text-slate-500"><button type="button" aria-label="Previous page" className="rounded border border-slate-700 p-1 disabled:opacity-40" disabled={safePage === 0} onClick={() => setPage((current) => Math.max(0, current - 1))}><ChevronLeft className="h-4 w-4" /></button><span>Page {safePage + 1} / {totalPages}</span><button type="button" aria-label="Next page" className="rounded border border-slate-700 p-1 disabled:opacity-40" disabled={safePage >= totalPages - 1} onClick={() => setPage((current) => Math.min(totalPages - 1, current + 1))}><ChevronRight className="h-4 w-4" /></button></div></div>
        <div className="max-h-72 overflow-auto"><table className="w-full min-w-155 text-left text-xs"><thead className="sticky top-0 bg-slate-900 text-[10px] uppercase tracking-wider text-slate-500"><tr><th className="px-4 py-2">Serial No.</th><th className="px-4 py-2">Reference (x, y)</th><th className="px-4 py-2">Source (x, y)</th><th className="px-4 py-2">Confidence</th></tr></thead><tbody className="divide-y divide-slate-800/70 text-slate-300">{visibleRows.map((row) => <tr key={row.id} className="hover:bg-slate-800/40"><td className="px-4 py-2 font-mono text-slate-500">{row.serialNumber}</td><td className="px-4 py-2 font-mono">{formatNumber(row.reference[0])}, {formatNumber(row.reference[1])}</td><td className="px-4 py-2 font-mono">{formatNumber(row.moving[0])}, {formatNumber(row.moving[1])}</td><td className="px-4 py-2 font-mono text-cyan-300">{formatNumber(row.confidence, 4)}</td></tr>)}</tbody></table>{!visibleRows.length && <p className="px-4 py-6 text-center text-xs text-slate-500">No inliers satisfy this confidence threshold.</p>}</div>
      </div>
    </section>
  );
}