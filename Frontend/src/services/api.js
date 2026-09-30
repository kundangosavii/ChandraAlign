import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Accept': 'application/json',
  }
});

/**
 * Perform Lunar Image Registration
 * @param {Object} files - The set of 4 required files
 * @param {File} files.sourceImg
 * @param {File} files.sourceXml
 * @param {File} files.refImg
 * @param {File} files.refXml
 * @returns {Promise<Object>} API response containing base64 images and metrics
 */
export const registerImages = async (
  { sourceImg, sourceXml, refImg, refXml }
) => {
  const formData = new FormData();
  formData.append('source_img', sourceImg);
  formData.append('source_xml', sourceXml);
  formData.append('ref_img', refImg);
  formData.append('ref_xml', refXml);

  const response = await apiClient.post('/image/match', formData, {
    // Let Axios set the multipart boundary automatically.
  });

  return response.data;
};

export const getExecutionLogs = async () => {
  const response = await apiClient.get('/image/logs');
  return response.data;
};

export const downloadExecutionLog = async (jobId) => {
  const response = await apiClient.get(`/image/logs/${jobId}/download`, {
    responseType: 'blob',
  });

  const url = window.URL.createObjectURL(response.data);
  const link = document.createElement('a');
  link.href = url;
  link.download = `${jobId}.csv`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.URL.revokeObjectURL(url);
};