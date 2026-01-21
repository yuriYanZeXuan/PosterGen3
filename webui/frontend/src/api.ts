import axios from 'axios';
import { JobStatus, PosterConfig, UploadedFiles } from './types';

const API_BASE = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE,
});

export const apiService = {
  async getModels(): Promise<string[]> {
    const response = await api.get('/models');
    return response.data.models;
  },

  async generatePoster(
    config: PosterConfig,
    files: UploadedFiles
  ): Promise<JobStatus> {
    const formData = new FormData();
    
    formData.append('text_model', config.text_model);
    formData.append('vision_model', config.vision_model);
    formData.append('poster_width', config.poster_width.toString());
    formData.append('poster_height', config.poster_height.toString());
    
    if (files.pdf_file) formData.append('pdf_file', files.pdf_file);
    if (files.logo_file) formData.append('logo_file', files.logo_file);
    if (files.aff_logo_file) formData.append('aff_logo_file', files.aff_logo_file);

    const response = await api.post('/generate', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    return response.data;
  },

  async getJobStatus(jobId: string): Promise<JobStatus> {
    const response = await api.get(`/status/${jobId}`);
    return response.data;
  },

  async getJobLogs(jobId: string): Promise<string[]> {
    const response = await api.get(`/logs/${jobId}`);
    return response.data.logs;
  },

  getDownloadUrl(jobId: string): string {
    return `${API_BASE}/download/${jobId}`;
  },
};