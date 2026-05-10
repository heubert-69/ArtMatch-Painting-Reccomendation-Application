<template>
  <div class="page">
    <!-- Navbar -->
    <nav class="navbar">
      <router-link to="/home">
        <img src="@/assets/LOGO1.png" alt="ArtMatch Logo" class="logo" />
      </router-link>
      <img src="@/assets/ACC_LOGO.png" alt="Profile" class="profile-icon" />
    </nav>

    <div class="edit-container">
      <h1 class="title">Profile</h1>

      <div class="content">
        <div class="sidebar">
          <button class="tab-btn" :class="{ active: activeTab === 'basic' }" @click="activeTab = 'basic'">Basic
            Information</button>
          <button class="tab-btn" :class="{ active: activeTab === 'security' }" @click="activeTab = 'security'">Account
            & Security</button>
        </div>

        <!-- Basic Information Tab -->
        <div class="form-section" v-if="activeTab === 'basic'">
          <!-- Avatar -->
          <div class="avatar-section">
            <img src="@/assets/ACC_LOGO.png" class="avatar" />
            <p class="replace-link">/ Replace</p>
          </div>

          <!-- Form Fields -->
          <div class="fields">
            <div class="row">
              <div class="form-group">
                <label>First Name</label>
                <input type="text" v-model="firstName" class="input" />
              </div>
              <div class="form-group">
                <label>Last Name</label>
                <input type="text" v-model="lastName" class="input" />
              </div>
            </div>

            <div class="form-group full">
              <label>Headline</label>
              <input type="text" v-model="headline" class="input" />
            </div>

            <div class="form-group">
              <label>Role</label>
              <select v-model="role" class="input select">
                <option value="">Select Role</option>
                <option value="artist">Artist</option>
                <option value="viewer">Viewer</option>
              </select>
            </div>

            <div class="form-group full">
              <label>About me</label>
              <textarea v-model="bio" class="textarea" placeholder="Tell us about yourself..."></textarea>
            </div>

            <!-- Links -->
            <div class="form-group full">
              <label>LINKS</label>
              <div class="links-table">
                <div class="links-header">
                  <span>Title</span>
                  <span>URL</span>
                  <span></span>
                </div>
                <div class="link-row" v-for="(link, index) in links" :key="index">
                  <input type="text" v-model="link.title" placeholder="Title" class="input link-input" />
                  <input type="text" v-model="link.url" placeholder="URL" class="input link-input" />
                  <button class="add-btn">Add</button>
                </div>
              </div>
            </div>

            <button class="save-btn" @click="saveBasicInfo">Save Changes</button>
          </div>
        </div>

        <!-- Account & Security Tab -->
        <div class="form-section" v-if="activeTab === 'security'">
          <div class="fields">
            <div class="form-group full">
              <label>Email</label>
              <input type="email" v-model="email" class="input" />
            </div>

            <div class="form-group full">
              <label>Username</label>
              <input type="text" v-model="username" class="input" />
            </div>

            <hr class="divider" />
            <p class="section-label">Change Password</p>

            <div class="form-group full">
              <label>Current Password</label>
              <input type="password" v-model="currentPassword" class="input" />
            </div>

            <div class="form-group full">
              <label>New Password</label>
              <input type="password" v-model="newPassword" class="input" />
            </div>

            <div class="form-group full">
              <label>Confirm New Password</label>
              <input type="password" v-model="confirmNewPassword" class="input" />
            </div>

            <p class="error-txt" v-if="errorMessage">{{ errorMessage }}</p>

            <button class="save-btn" @click="saveSecurityInfo">Save Changes</button>

            <hr class="divider" />
            <button class="delete-btn">Delete Account</button>
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
const activeTab = ref('basic')

// Basic Info
const firstName = ref('')
const lastName = ref('')
const headline = ref('')
const role = ref('')
const bio = ref('')
const links = ref([
  { title: '', url: '' },
  { title: '', url: '' }
])

// Security
const email = ref('')
const username = ref('')
const currentPassword = ref('')
const newPassword = ref('')
const confirmNewPassword = ref('')
const errorMessage = ref('')

const saveBasicInfo = () => {
  console.log('Saving basic info...')
  // API call goes here later
}

const saveSecurityInfo = () => {
  if (newPassword.value !== confirmNewPassword.value) {
    errorMessage.value = 'Passwords do not match!'
    return
  }
  errorMessage.value = ''
  console.log('Saving security info...')
  // API call goes here later
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

.edit-container {
  max-width: 800px;
  margin: 30px auto;
  background: white;
  border-radius: 16px;
  padding: 30px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
}

.title {
  font-family: 'ClimateCrisis', sans-serif;
  font-size: 3rem;
  font-weight: 900;
  color: #1a1a2e;
  text-align: center;
  margin-bottom: 30px;
}

.content {
  display: flex;
  gap: 24px;
}

.sidebar {
  display: flex;
  flex-direction: column;
  gap: 8px;
  min-width: 160px;
}

.tab-btn {
  padding: 12px 16px;
  background: #f0f0f0;
  border: none;
  border-radius: 8px;
  font-family: 'JosefinSans', sans-serif;
  font-size: 0.85rem;
  font-weight: 700;
  cursor: pointer;
  text-align: left;
  transition: background 0.2s;
}

.tab-btn.active {
  background: #1a1a2e;
  color: white;
}

.tab-btn:hover:not(.active) {
  background: #ddd;
}

.form-section {
  display: flex;
  gap: 24px;
  flex: 1;
}

.avatar-section {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
}

.avatar {
  width: 80px;
  height: 80px;
  border-radius: 50%;
  object-fit: cover;
  opacity: 0.5;
  background: #d9d9d9;
}

.replace-link {
  font-family: 'JosefinSans', sans-serif;
  font-size: 0.8rem;
  color: #FE5D26;
  cursor: pointer;
}

.fields {
  display: flex;
  flex-direction: column;
  gap: 14px;
  flex: 1;
}

.row {
  display: flex;
  gap: 12px;
}

.form-group {
  display: flex;
  flex-direction: column;
  gap: 4px;
  flex: 1;
}

.form-group.full {
  width: 100%;
}

.form-group label {
  font-family: 'JosefinSans', sans-serif;
  font-weight: 700;
  font-size: 0.85rem;
  color: #1a1a2e;
}

.input {
  font-family: 'JosefinSans', sans-serif;
  border: 1.5px solid #ccc;
  border-radius: 6px;
  padding: 8px 12px;
  font-size: 0.85rem;
  outline: none;
  color: #1a1a2e;
}

.input:focus {
  border-color: #FE5D26;
}

.select {
  cursor: pointer;
}

.textarea {
  font-family: 'JosefinSans', sans-serif;
  border: 1.5px solid #ccc;
  border-radius: 6px;
  padding: 10px 12px;
  font-size: 0.85rem;
  outline: none;
  color: #1a1a2e;
  height: 120px;
  resize: vertical;
}

.textarea:focus {
  border-color: #FE5D26;
}

.links-table {
  border: 1.5px solid #ccc;
  border-radius: 6px;
  overflow: hidden;
}

.links-header {
  display: grid;
  grid-template-columns: 1fr 2fr auto;
  padding: 8px 12px;
  background: #f9f9f9;
  font-family: 'JosefinSans', sans-serif;
  font-weight: 700;
  font-size: 0.8rem;
  color: #1a1a2e;
  border-bottom: 1px solid #ccc;
}

.link-row {
  display: grid;
  grid-template-columns: 1fr 2fr auto;
  align-items: center;
  padding: 6px 8px;
  gap: 8px;
  border-bottom: 1px solid #eee;
}

.link-input {
  border: 1px solid #eee;
  border-radius: 4px;
  padding: 6px 8px;
}

.add-btn {
  padding: 6px 14px;
  background: #1a1a2e;
  color: white;
  border: none;
  border-radius: 6px;
  font-family: 'JosefinSans', sans-serif;
  font-size: 0.8rem;
  cursor: pointer;
}

.add-btn:hover {
  background: #FE5D26;
}

.save-btn {
  width: fit-content;
  padding: 10px 24px;
  background: #1a1a2e;
  color: white;
  border: none;
  border-radius: 50px;
  font-family: 'JosefinSans', sans-serif;
  font-size: 0.85rem;
  font-weight: 700;
  cursor: pointer;
  transition: background 0.2s;
}

.save-btn:hover {
  background: #FE5D26;
}

.divider {
  border: none;
  border-top: 1px solid #eee;
  margin: 8px 0;
}

.section-label {
  font-family: 'JosefinSans', sans-serif;
  font-weight: 700;
  font-size: 1rem;
  color: #1a1a2e;
}

.error-txt {
  font-family: 'JosefinSans', sans-serif;
  font-size: 0.85rem;
  color: #ff0000;
  font-weight: 700;
}

.delete-btn {
  width: fit-content;
  padding: 10px 24px;
  background: transparent;
  color: #ff0000;
  border: 1.5px solid #ff0000;
  border-radius: 50px;
  font-family: 'JosefinSans', sans-serif;
  font-size: 0.85rem;
  font-weight: 700;
  cursor: pointer;
  transition: all 0.2s;
}

.delete-btn:hover {
  background: #ff0000;
  color: white;
}
</style>