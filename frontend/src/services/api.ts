import axios from 'axios'

const API_BASE_URL = 'http://127.0.0.1:8000/api'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

export const uploadMedia = async (file: File, model: string) => {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('model', model)

  const response = await apiClient.post('/analyze', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })

  return response.data
}

export const checkHealth = async () => {
  const response = await apiClient.get('/health')
  return response.data
}

export default apiClient
