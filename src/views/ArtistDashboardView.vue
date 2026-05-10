<template>
  <div class="page">
    <!-- Navbar -->
    <nav class="navbar">
          <router-link to="/home">
            <img src="@/assets/LOGO1.png" alt="ArtMatch Logo" class="logo" />
          </router-link>         
        <img src="@/assets/ACC_LOGO.png" alt="Profile" class="profile-icon" />
    </nav>

    <div class="dashboard">
      <!-- Profile Section -->
      <div class="profile-section">
        <div class="profile-card">
          <img src="@/assets/ACC_LOGO.png" class="avatar" />
          <h2 class="username">[Username]</h2>
          <p class="headline">[Headline]</p>

          <div class="stats">
            <div class="stat">
              <span class="stat-number">[n]</span>
              <span class="stat-label">Following</span>
            </div>
            <div class="stat">
              <span class="stat-number">[n]</span>
              <span class="stat-label">Followers</span>
            </div>
            <div class="stat">
              <span class="stat-number">[n]</span>
              <span class="stat-label">Appreciations</span>
            </div>
          </div>

          <p class="bio">[Bio]</p>
          <a href="#" class="link">[Link]</a>
          <button class="edit-btn" @click="router.push('/edit-profile')">Edit Profile</button>
        </div>  <!-- profile-card closes HERE -->
      </div>  <!-- profile-section closes HERE -->

      <!-- Paintings Grid -->
      <div class="grid-section">
        <div class="grid">
          <div
            class="card"
            v-for="(painting, index) in paintings"
            :key="index"
          >
            <div class="card-image"></div>
            <div class="card-info">
              <p class="card-title">{{ painting.title }}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
const router = useRouter()

const paintings = ref([
  ...Array(12).fill({ title: 'Title', artist: 'Artist' })
])
</script>

<style scoped>
.page {
  background-color: #f5f5f5;
  min-height: 100vh;
}

.navbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 24px;
  background: white;
  border-bottom: 1px solid #eee;
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

.dashboard {
  max-width: 1100px;
  margin: 30px auto;
  background: white;
  border-radius: 16px;
  overflow: hidden;
  box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}

.profile-section {
  display: flex;
  justify-content: center;
  padding: 40px 20px;
  border-bottom: 1px solid #eee;
}

.profile-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  text-align: center;
}

.avatar {
  width: 90px;
  height: 90px;
  border-radius: 50%;
  object-fit: cover;
  opacity: 0.5;
  background: #d9d9d9;
}

.username {
  font-family: 'ClimateCrisis', sans-serif;
  font-size: 1.5rem;
  color: #1a1a2e;
}

.headline {
  font-family: 'JosefinSans', sans-serif;
  font-size: 0.95rem;
  color: #555;
}

.bio {
  font-family: 'JosefinSans', sans-serif;
  font-size: 0.85rem;
  color: #888;
  max-width: 400px;
}

.link {
  font-family: 'JosefinSans', sans-serif;
  font-size: 0.85rem;
  color: #FE5D26;
  text-decoration: none;
}

.link:hover {
  text-decoration: underline;
}

.grid-section {
  padding: 20px;
}

.grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 2px;
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
  height: 140px;
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

.stats {
  display: flex;
  gap: 24px;
  margin: 8px 0;
}

.stat {
  display: flex;
  flex-direction: row;
  align-items: center;
  gap: 4px;
  font-family: 'JosefinSans', sans-serif;
  font-size: 0.85rem;
  color: #1a1a2e;
}

.stat-number {
  font-weight: 700;
}

.edit-btn {
  margin-top: 8px;
  padding: 10px 24px;
  background-color: #1a1a2e;
  color: white;
  border: none;
  border-radius: 50px;
  font-family: 'JosefinSans', sans-serif;
  font-size: 0.85rem;
  font-weight: 700;
  cursor: pointer;
  transition: background 0.2s;
}

.edit-btn:hover {
  background-color: #FE5D26;
}

</style>