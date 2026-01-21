import React, { useState, useEffect } from 'react';
import { FileUpload } from './components/FileUpload';
import { ProgressBar } from './components/ProgressBar';
import { apiService } from './api';
import { PosterConfig, UploadedFiles, JobStatus } from './types';
import postergenLogo from './postergen-logo.png';

function App() {
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [config, setConfig] = useState<PosterConfig>({
    text_model: '',
    vision_model: '',
    poster_width: 54,
    poster_height: 36,
  });
  const [files, setFiles] = useState<UploadedFiles>({
    pdf_file: null,
    logo_file: null,
    aff_logo_file: null,
  });
  const [currentJob, setCurrentJob] = useState<JobStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [jsonFiles, setJsonFiles] = useState<Record<string, any>>({});
  const [expandedFiles, setExpandedFiles] = useState<Set<string>>(new Set());

  useEffect(() => {
    const loadModels = async () => {
      try {
        const models = await apiService.getModels();
        setAvailableModels(models);
        if (models.length > 0) {
          setConfig(prev => ({
            ...prev,
            text_model: models[0],
            vision_model: models[0],
          }));
        }
      } catch (err) {
        setError('Failed to load available models');
      }
    };
    
    loadModels();
  }, []);

  useEffect(() => {
    if (!currentJob || currentJob.status === 'completed' || currentJob.status === 'failed') {
      return;
    }

    const pollInterval = setInterval(async () => {
      try {
        const [status, logs] = await Promise.all([
          apiService.getJobStatus(currentJob.job_id),
          apiService.getJobLogs(currentJob.job_id)
        ]);
        
        setCurrentJob(status);
        setLogs(logs);
        
        if (status.status === 'failed') {
          setError(status.error || 'Job failed');
          setIsSubmitting(false);
        } else if (status.status === 'completed') {
          setIsSubmitting(false);
          fetchJsonFiles(currentJob.job_id);
        }
      } catch (err) {
        setError('Failed to check job status');
        setIsSubmitting(false);
      }
    }, 2000);

    return () => clearInterval(pollInterval);
  }, [currentJob]);

  const fetchJsonFiles = async (jobId: string) => {
    try {
      const response = await fetch(`http://localhost:8000/files/${jobId}`);
      if (response.ok) {
        const data = await response.json();
        setJsonFiles(data.files || {});
      }
    } catch (error) {
      console.error('Failed to fetch JSON files:', error);
    }
  };

  const getPosterImageUrl = () => {
    if (!currentJob || currentJob.status !== 'completed') return null;
    return `http://localhost:8000/poster/${currentJob.job_id}`;
  };

  const toggleFileExpansion = (filename: string) => {
    const newExpanded = new Set(expandedFiles);
    if (newExpanded.has(filename)) {
      newExpanded.delete(filename);
    } else {
      newExpanded.add(filename);
    }
    setExpandedFiles(newExpanded);
  };

  const copyToClipboard = (content: string) => {
    navigator.clipboard.writeText(content);
  };

  const handleConfigChange = (field: keyof PosterConfig, value: string | number) => {
    setConfig(prev => ({ ...prev, [field]: value }));
  };

  const handleFileChange = (field: keyof UploadedFiles, file: File) => {
    setFiles(prev => ({ ...prev, [field]: file }));
    setError(null);
  };

  const validateForm = (): string | null => {
    if (!files.pdf_file) return 'Please upload a PDF paper';
    if (!files.logo_file) return 'Please upload a conference logo';
    if (!files.aff_logo_file) return 'Please upload an affiliation logo';
    
    const ratio = config.poster_width / config.poster_height;
    if (ratio < 1.4 || ratio > 2.0) {
      return `Poster ratio ${ratio.toFixed(2)} is out of range (1.4 - 2.0)`;
    }
    
    return null;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    const validationError = validateForm();
    if (validationError) {
      setError(validationError);
      return;
    }

    setIsSubmitting(true);
    setError(null);
    setCurrentJob(null);

    try {
      const jobStatus = await apiService.generatePoster(config, files);
      setCurrentJob(jobStatus);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to start poster generation');
      setIsSubmitting(false);
    }
  };

  const canSubmit = files.pdf_file && files.logo_file && files.aff_logo_file && !isSubmitting;

  return (
    <div className="container">
      <div className="header">
        <h1 style={{display: 'flex', alignItems: 'center', justifyContent: 'center'}}><img src={postergenLogo} alt="PosterGen Logo" style={{height: '1.5em', marginRight: '0.5em'}} />PosterGen WebUI</h1>
        <p>🎨 Generate design-aware academic posters from PDF papers</p>
      </div>

      <form onSubmit={handleSubmit} className="main-form">
        <div className="form-section">
          <h3 className="section-title">📄 Upload Files</h3>
          
          <div className="form-group">
            <label>PDF Paper</label>
            <FileUpload
              label="PDF Paper"
              accept="application/pdf"
              selectedFile={files.pdf_file}
              onFileSelect={(file) => handleFileChange('pdf_file', file)}
            />
          </div>

          <div className="form-row">
            <div className="form-group">
              <label>Conference Logo</label>
              <FileUpload
                label="Logo"
                accept="image/*"
                selectedFile={files.logo_file}
                onFileSelect={(file) => handleFileChange('logo_file', file)}
              />
            </div>
            
            <div className="form-group">
              <label>Affiliation Logo</label>
              <FileUpload
                label="Affiliation Logo"
                accept="image/*"
                selectedFile={files.aff_logo_file}
                onFileSelect={(file) => handleFileChange('aff_logo_file', file)}
              />
            </div>
          </div>
        </div>

        <div className="form-section">
          <h3 className="section-title">🤖 Model Configuration</h3>
          
          <div className="form-row">
            <div className="form-group">
              <label>Text Model</label>
              <select
                value={config.text_model}
                onChange={(e) => handleConfigChange('text_model', e.target.value)}
              >
                {availableModels.map(model => (
                  <option key={model} value={model}>{model}</option>
                ))}
              </select>
            </div>
            
            <div className="form-group">
              <label>Vision Model</label>
              <select
                value={config.vision_model}
                onChange={(e) => handleConfigChange('vision_model', e.target.value)}
              >
                {availableModels.map(model => (
                  <option key={model} value={model}>{model}</option>
                ))}
              </select>
            </div>
          </div>
        </div>

        <div className="form-section">
          <h3 className="section-title">📐 Poster Dimensions</h3>
          
          <div className="form-row">
            <div className="form-group">
              <label>Width (inches)</label>
              <input
                type="number"
                min="20"
                max="100"
                step="0.1"
                value={config.poster_width}
                onChange={(e) => handleConfigChange('poster_width', parseFloat(e.target.value) || 54)}
              />
            </div>
            
            <div className="form-group">
              <label>Height (inches)</label>
              <input
                type="number"
                min="10"
                max="60"
                step="0.1"
                value={config.poster_height}
                onChange={(e) => handleConfigChange('poster_height', parseFloat(e.target.value) || 36)}
              />
            </div>
          </div>
          
        </div>

        <button
          type="submit"
          className="button"
          disabled={!canSubmit}
        >
          {isSubmitting ? 'Generating Poster...' : 'Generate Poster'}
        </button>

        {error && (
          <div className="error-message">
            {error}
          </div>
        )}

        {currentJob && currentJob.status !== 'failed' && (
          <ProgressBar
            message={currentJob.message}
            logs={logs}
            isActive={currentJob.status === 'processing' || currentJob.status === 'pending'}
          />
        )}

        {currentJob && currentJob.status === 'completed' && (
          <div className="download-section">
            <div className="success-message">
              Poster generation completed successfully!
            </div>
            <a
              href={apiService.getDownloadUrl(currentJob.job_id)}
              className="download-button"
              download
            >
              Download Poster Files
            </a>
          </div>
        )}
      </form>

      <div className="preview-wrapper">
        <div className="preview-section">
          <h3 className="section-title">📄 Meta Files</h3>
          <div className="json-viewer">
            {Object.keys(jsonFiles).length > 0 ? (
              Object.entries(jsonFiles).map(([filename, content]) => (
                <div key={filename} className="json-file">
                  <div
                    className="json-file-header"
                    onClick={() => toggleFileExpansion(filename)}
                  >
                    <span>{filename}</span>
                    <div>
                      <button
                        className="copy-btn"
                        onClick={(e) => {
                          e.stopPropagation();
                          copyToClipboard(JSON.stringify(content, null, 2));
                        }}
                      >
                        Copy
                      </button>
                      <span style={{ marginLeft: '8px' }}>
                        {expandedFiles.has(filename) ? '−' : '+'}
                      </span>
                    </div>
                  </div>
                  {expandedFiles.has(filename) && (
                    <div className="json-content">
                      <pre>{JSON.stringify(content, null, 2)}</pre>
                    </div>
                  )}
                </div>
              ))
            ) : (
              <div className="empty-state">
                {currentJob?.status === 'completed' 
                  ? 'No files available'
                  : 'JSON files will appear after generation'
                }
              </div>
            )}
          </div>
        </div>
        
        <div className="section-divider"></div>
        
        <div className="preview-section">
          <h3 className="section-title">🖼️ Poster Preview</h3>
          <div className="preview-content">
            {currentJob?.status === 'completed' && getPosterImageUrl() ? (
              <div className="preview-container">
                <img
                  src={getPosterImageUrl()!}
                  alt="Generated Poster"
                  className="poster-preview"
                />
              </div>
            ) : (
              <div className="empty-state">
                {currentJob?.status === 'processing' || currentJob?.status === 'pending' 
                  ? 'Preview will appear when generation is complete...'
                  : 'Upload files and generate a poster to see preview'
                }
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;