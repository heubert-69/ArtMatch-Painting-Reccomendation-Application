<template>
  <div class="page">
    <!-- Navbar -->
    <nav class="navbar">
      <router-link to="/home">
        <img src="@/assets/LOGO1.png" alt="ArtMatch Logo" class="logo" />
      </router-link>        
      <img src="@/assets/ACC_LOGO.png" alt="Profile" class="profile-icon" />
    </nav>

    <div class="post-container">
      <!-- Toolbar -->
      <div class="toolbar">
        <div class="tool-item">
        <img src = "@/assets/Image-logo.png" class="tool-icon-img"/>
        <p class="tool-label">Add Image</p>
        </div>

        <div class="tool-item">
        <img src = "@/assets/Text-Logo.png" class="tool-icon-img"/>
        <p class="tool-label">Add Text</p>
        </div>

        <div class="tool-item">
          <img src="@/assets/URL-Logo.png" class="tool-icon-img"/>
          <p class="tool-label">Add Link</p>
        </div>
        <div class="tool-item">
          <img src="@/assets/Div-Logo.png" class ="tool-icon-img"/>
          <p class="tool-label">Add Division</p>
        </div>
      </div>

      <!-- Project Cover -->
      <div class="cover-section">
        <p class="cover-label">Project Cover</p>
        <div class="cover-box" @click="triggerImageUpload">
          <input 
            type="file" 
            ref="imageInput" 
            accept="image/*" 
            style="display:none" 
            @change="handleImageUpload"
          />
          <img v-if="coverImage" :src="coverImage" class="cover-preview" />
          <div v-else class="cover-placeholder">
            <span>✥</span>
            <p class="cover-title">Title</p>
            <p class="cover-artist">Artist</p>
          </div>
        </div>
      </div>

      <!-- Form -->
      <div class="form-group">
        <label>Title <span class="required">(Required)</span></label>
        <input type="text" placeholder="Title" v-model="title" class="input" />
      </div>

      <div class="form-group">
        <label>Tags</label>
        <input type="text" placeholder="Tags" v-model="tags" class="input" />
      </div>

      <button class="btn" @click="submitPost">Post</button>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()
const title = ref('')
const tags = ref('')
const coverImage = ref(null)
const imageInput = ref(null)

const triggerImageUpload = () => {
  imageInput.value.click()
}

const handleImageUpload = (e) => {
  const file = e.target.files[0]
  if (file) {
    coverImage.value = URL.createObjectURL(file)
  }
}

const submitPost = () => {
  if (!title.value) {
    alert('Title is required!')
    return
  }
  console.log('Posting:', title.value, tags.value, coverImage.value)
  // API call goes here later
  router.push('/artist-dashboard')
}
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

.post-container {
  max-width: 800px;
  margin: 30px auto;
  background: white;
  border-radius: 16px;
  padding: 30px;
  box-shadow: 0 4px 20px rgba(0,0,0,0.08);
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.toolbar {
  display: flex;
  justify-content: center;
  gap: 40px;
  padding: 16px;
  border: 1.5px solid #eee;
  border-radius: 12px;
}

.tool-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
  cursor: pointer;
}

.tool-icon {
  width: 56px;
  height: 56px;
  border-radius: 50%;
  background: #e0e0e0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.2rem;
  font-weight: 900;
  transition: background 0.2s;
}

.tool-icon:hover {
  background: #FE5D26;
  color: white;
}

.tool-icon-img {
  width: 56px;
  height: 56px;
  object-fit: contain;
  border-radius: 50%;
  cursor: pointer;
  transition: opacity 0.2s;
}

.tool-icon-img:hover {
  opacity: 0.7;
}
.tool-label {
  font-family: 'JosefinSans', sans-serif;
  font-size: 0.75rem;
  color: #555;
}

.text-icon{
  font-family: 'ClimateCrisis', sans-serif;
  font-size: 1.5rem;
}

.cover-label {
  font-family: 'JosefinSans', sans-serif;
  font-weight: 700;
  font-size: 0.95rem;
  color: #1a1a2e;
  margin-bottom: 8px;
}

.cover-box {
  width: 200px;
  height: 150px;
  border: 1.5px solid #ccc;
  border-radius: 8px;
  cursor: pointer;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #f9f9f9;
}

.cover-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  color: #aaa;
}

.cover-title {
  font-family: 'JosefinSans', sans-serif;
  font-weight: 700;
  font-size: 0.85rem;
  color: #1a1a2e;
}

.cover-artist {
  font-family: 'JosefinSans', sans-serif;
  font-size: 0.75rem;
  color: #888;
}

.cover-preview {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.form-group {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.form-group label {
  font-family: 'JosefinSans', sans-serif;
  font-weight: 700;
  font-size: 0.95rem;
  color: #1a1a2e;
}

.required {
  font-weight: 400;
  color: #FE5D26;
}

.input {
  font-family: 'JosefinSans', sans-serif;
  border: 1.5px solid #ccc;
  border-radius: 8px;
  padding: 10px 14px;
  font-size: 0.9rem;
  outline: none;
  width: 320px;
  color: #1a1a2e;
}

.input:focus {
  border-color: #FE5D26;
}

.btn {
  width: 100px;
  padding: 12px;
  background-color: #1a1a2e;
  color: white;
  border: none;
  border-radius: 50px;
  font-size: 0.85rem;
  font-weight: 700;
  letter-spacing: 2px;
  cursor: pointer;
  transition: background 0.2s;
}

.btn:hover {
  background-color: #FE5D26;
}
</style>