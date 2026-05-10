import axios from 'axios'

const api = axios.create({
  baseURL: 'http://127.0.0.1:5000/api', // your Flask backend URL
  headers: {
    'Content-Type': 'application/json'
  }
})

export const authAPI = {
  login: (credentials) => api.post('/login', credentials),
  register: (userData) => api.post('/register', userData),
  logout: () => api.post('/logout')
}

// Paintings endpoints
export const paintingsAPI = {
  getAll: () => api.get('/paintings'),
  getOne: (id) => api.get(`/paintings/${id}`),
  create: (data) => api.post('/paintings', data),
  update: (id, data) => api.put(`/paintings/${id}`, data),
  delete: (id) => api.delete(`/paintings/${id}`),
  recommend: (data) => api.post('/recommend', data)
}

// User endpoints
export const userAPI = {
  getProfile: (id) => api.get(`/users/${id}`),
  updateProfile: (id, data) => api.put(`/users/${id}`, data)
}

export default api
