<template>
  <div class="page">
    <div class="hero">
      <video class="bg-video" autoplay muted loop playsinline>
        <source src="@/assets/BG.mp4" type="video/mp4" />
      </video>

      <!-- Navbar -->
      <nav class="navbar" :class="{ scrolled: scrolled }">
        <router-link to="/home">
          <img src="@/assets/LOGO1.png" alt="ArtMatch Logo" class="logo" />
        </router-link>        
    <div class="profile-wrapper">
  <img 
    src="@/assets/ACC_LOGO.png" 
    alt="Profile" 
    class="profile-icon" 
    @click="toggleDropdown"
  />
  <div class="dropdown" v-if="showDropdown">
    <div class="dropdown-header">
      <img src="@/assets/ACC_LOGO.png" class="dropdown-avatar" />
      <div>
        <p class="dropdown-name">{{ authStore.user?.username || 'Name' }}</p>
        <p class="dropdown-role">{{ authStore.role || '[Role Viewer/Artist]' }}</p>
      </div>
    </div>
    <hr />
    <p class="dropdown-item" @click="router.push('/artist-dashboard')">Profile</p>    
    <p class="dropdown-item" @click="router.push('/edit-profile')">Settings</p>
    <p class="dropdown-item" @click="router.push('/')">Sign Out</p>
  </div>
</div>
      </nav>

      <!-- Header -->
      <div class="header">
        <h1 class="brand">ArtMatch</h1>
          <div class="search-wrapper">
          <input type="text" placeholder="Search" v-model="search" class="search-input" />
          <span class="search-icon">🔍</span>
          </div>
        <p class="section-title">Recommended for you</p>
      </div>
    </div>

    <!-- Painting Grid -->
    <div class="grid">
      <div
        class="card"
        v-for="(painting, index) in paintings"
        :key="index"
        @click="goToDetail(painting)"
      >
        <div class="card-image"></div>
        <div class="card-info">
          <p class="card-title">{{ painting.title }}</p>
          <p class="card-artist">{{ painting.artist }}</p>
        </div>
      </div>
    </div>
             <button class="fab" @click="goToPost">+</button>
  </div>
</template>

<script setup>
import { useRouter } from 'vue-router'
import { ref, onMounted, onUnmounted } from 'vue'
import { useAuthStore } from '@/stores/auth'

const authStore = useAuthStore()
const scrolled = ref(false)

const handleScroll = () => {
  scrolled.value = window.scrollY > 50
}

onMounted(() => window.addEventListener('scroll', handleScroll))
onUnmounted(() => window.removeEventListener('scroll', handleScroll))

const router = useRouter()
const search = ref('')

const goToPost = () => {
  router.push('/post')
}

const showDropdown = ref(false)
const toggleDropdown = () => {
  showDropdown.value = !showDropdown.value
}
// Placeholder data — will be replaced by API later
const paintings = ref([
  { title: 'Title', artist: 'Artist' },
  { title: 'Title', artist: 'Artist' },
  { title: 'Title', artist: 'Artist' },
  { title: 'Title', artist: 'Artist' },
  { title: 'Title', artist: 'Artist' },
  { title: 'Title', artist: 'Artist' },
  { title: 'Title', artist: 'Artist' },
  { title: 'Title', artist: 'Artist' },
  { title: 'Title', artist: 'Artist' },
  { title: 'Title', artist: 'Artist' },
  { title: 'Title', artist: 'Artist' },
  { title: 'Title', artist: 'Artist' },
  { title: 'Title', artist: 'Artist' },
  { title: 'Title', artist: 'Artist' },
  { title: 'Title', artist: 'Artist' },
  { title: 'Title', artist: 'Artist' },
])

const goToDetail = (painting) => {
  router.push('/painting-detail')
}
</script>

<style scoped>

.hero {
  position: relative;
  overflow: hidden;
  height: 350px;
  margin-bottom: 40px;
}

.bg-video {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
  z-index: 0;
}

.navbar, .header {
  position: relative;
  z-index: 1;
}
.navbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 24px;
  border-bottom: 1px solid #eee;
  position: sticky;
  top: 0;
  background: white;
  z-index: 10;
  transition: all 0.3s ease;
  backdrop-filter: blur(8px);
}

.profile-wrapper {
  position: relative;
}

.dropdown {
  position: absolute;
  right: 0;
  top: 50px;
  background: white;
  border: 1px solid #eee;
  border-radius: 12px;
  width: 200px;
  box-shadow: 0 8px 24px rgba(0,0,0,0.1);
  z-index: 100;
  overflow: hidden;
}

.dropdown-header {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 12px 16px;
  background: #f9f9f9;
}

.dropdown-avatar {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  opacity: 0.5;
}

.dropdown-name {
  font-family: 'JosefinSans', sans-serif;
  font-weight: 700;
  font-size: 0.9rem;
  color: #1a1a2e;
}

.dropdown-role {
  font-family: 'JosefinSans', sans-serif;
  font-size: 0.75rem;
  color: #888;
}

.dropdown-item {
  padding: 12px 16px;
  font-family: 'JosefinSans', sans-serif;
  font-size: 0.9rem;
  color: #1a1a2e;
  cursor: pointer;
  transition: background 0.2s;
}

.dropdown-item:hover {
  background: #f5f5f5;
  color: #FE5D26;
}

hr {
  border: none;
  border-top: 1px solid #eee;
  margin: 0;
}

.logo {
  width: 40px;
  height: 40px;
  object-fit: contain;
}

.profile-icon {
  width: 36px;
  height: 36px;
  object-fit: contain;
  border-radius: 50%;
  cursor: pointer;
  opacity: 0.5;
}

.header {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 20px;
  gap: 12px;
  position:relative;
  z-index: 1;
}

.brand {
  font-family: 'ClimateCrisis', sans-serif;
  font-size: 7rem;
  font-weight: 900;
  color: #FE5D26;
  transition: font-size 0.3s ease;
  filter: drop-shadow(0px 4px 10px rgba(255, 255, 255, 0.3));
}

.brand-small{
  font-size: 1.2rem;
}

.search-wrapper {
  display: flex;
  align-items: center;
  border: 1.5px solid #ccc;
  border-radius: 50px;
  padding: 8px 16px;
  width: 400px;
  gap: 8px;
  background: white;
}

.fab {
  position: fixed;
  bottom: 30px;
  left: 50%;
  transform: translateX(-50%);
  width: 56px;
  height: 56px;
  border-radius: 50%;
  background-color: #FE5D26;
  color: white;
  font-size: 2rem;
  border: none;
  cursor: pointer;
  box-shadow: 0 2px 8px rgba(94, 92, 92, 0.3);
  transition: background 0.2s;
  z-index: 100;
}

.fab:hover {
  background-color: #FE5D26;
}

.search-input {
  border: none;
  outline: none;
  width: 100%;
  font-family: 'JosefinSans', sans-serif;
  font-size: 0.9rem;
  color: #1a1a2e;
}

.search-icon {
  font-size: 0.9rem;
  color: #aaa;
}

.section-title {
  font-family: 'JosefinSans', sans-serif;
  font-weight: 700;
  font-size: 1rem;
  color: #1a1a2e;
}

.grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 2px;
  padding: 0 24px 24px;
}

.card {
  cursor: pointer;
  background: #f0f0f0;
  transition: transform 0.2s;
}

.card:hover {
  transform: scale(1.02);
}

.card-image {
  width: 100%;
  height: 160px;
  background-color: #d9d9d9;
}

.card-info {
  padding: 8px;
}

.card-title {
  font-family: 'JosefinSans', sans-serif;
  font-weight: 700;
  font-size: 0.85rem;
  color: #1a1a2e;
}

.card-artist {
  font-family: 'JosefinSans', sans-serif;
  font-size: 0.75rem;
  color: #888;
}
</style>