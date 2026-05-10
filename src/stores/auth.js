import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useAuthStore = defineStore('auth', () => {
  const user = ref(null)
  const isLoggedIn = ref(false)
  const role = ref('') // 'artist' or 'viewer'

  const login = (userData) => {
    user.value = userData
    isLoggedIn.value = true
    role.value = userData.role
  }

  const logout = () => {
    user.value = null
    isLoggedIn.value = false
    role.value = ''
  }

  const isArtist = () => role.value === 'artist'

  return { user, isLoggedIn, role, login, logout, isArtist }
})