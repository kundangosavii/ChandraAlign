import { useState } from 'react';
import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';
import UploadPanel from './components/UploadPanel';
import ExecuteButton from './components/ExecuteButton';
import ResultPanel from './components/ResultPanel';
import Loader from './components/Loader';
import { registerImages } from './services/api';
import { AlertTriangle } from 'lucide-react';

// Import newly created pages
import AlgorithmSettings from './pages/AlgorithmSettings';
import ExecutionLogs from './pages/ExecutionLogs';
import CalibratedDatasets from './pages/CalibratedDatasets';
import ApiReference from './pages/ApiReference';
import HelpDocs from './pages/HelpDocs';

export default function App() {
  const [activeTab, setActiveTab] = useState('registration');

  // Registration state
  const [files, setFiles] = useState({
    sourceImg: null,
    sourceXml: null,
    refImg: null,
    refXml: null,
  });
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const isFormValid = Boolean(
    files.sourceImg && files.sourceXml && files.refImg && files.refXml
  );

  const handleExecute = async () => {
    if (!isFormValid) return;
    setIsLoading(true);
    setError(null);

    try {
      const data = await registerImages(files);
      setResults(data);
    } catch (err) {
      console.error('API Error:', err);
      setError(
        err.response?.data?.detail ||
          err.message ||
          'Failed to communicate with FastAPI backend. Ensure server is running at http://127.0.0.1:8000'
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-950 text-slate-100 font-sans">
      <Navbar />

      <div className="flex flex-1 overflow-hidden">
        <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />

        <main className="flex-1 overflow-y-auto p-6 lg:p-8">
          {/* Main Module Route Views */}
          {activeTab === 'registration' && (
            <div className="space-y-6">
              <div className="border-b border-slate-800 pb-5">
                <h2 className="text-xl font-bold text-slate-100 tracking-tight">
                  Lunar Surface Image Registration
                </h2>
                <p className="text-xs text-slate-400 mt-1">
                  Lunar-Imagery surface alignment with detected correspondences and computed metrics.
                </p>
              </div>

              <UploadPanel files={files} setFiles={setFiles} />


              <ExecuteButton
                onExecute={handleExecute}
                isDisabled={!isFormValid}
                isLoading={isLoading}
              />

              {error && (
                <div className="p-4 bg-red-950/40 border border-red-800/60 rounded-xl flex items-start space-x-3 text-red-300">
                  <AlertTriangle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
                  <div className="text-xs space-y-1">
                    <span className="font-semibold text-red-200">Execution Error</span>
                    <p>{error}</p>
                  </div>
                </div>
              )}

              {isLoading && <Loader />}
              {!isLoading && results && (
                <ResultPanel
                  key={results.match_image_pages?.[0] || results.match_image}
                  results={results}
                />
              )}
            </div>
          )}

          {activeTab === 'settings' && <AlgorithmSettings />}
          {activeTab === 'logs' && <ExecutionLogs />}
          {activeTab === 'datasets' && <CalibratedDatasets />}
          {activeTab === 'api' && <ApiReference />}
          {activeTab === 'help' && <HelpDocs />}
        </main>
      </div>
    </div>
  );
}