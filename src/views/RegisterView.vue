<template>
  <div class="page">  
    <video class="bg-video" autoplay muted loop playsinline>
      <source src="@/assets/BG.mp4" type="video/mp4" />
    </video>

    <img src="@/assets/LOGO1.png" alt="ArtMatch Logo" class="logo" />

    <div class="container">
      <div class="glass-panel">
      <h1 class="title">SIGN UP</h1>
      <p class = "error-txt" v-if="errorMessage">{{ errorMessage }} </p>

      <div class="form-group">
        <label>Username</label>
        <div class="input-wrapper">
        <img src="@/assets/ACC_LOGO.png" class="icon" />          
        <input type="text" placeholder="Create Username" v-model="username" />
        </div>
      </div>

      <div class="form-group">
        <label>Email</label>
        <div class="input-wrapper">
        <img src="@/assets/ACC_LOGO.png" class="icon" />          
        <input type="text" placeholder="Enter Email" v-model="email" />
        </div>
      </div>

      <div class="form-group">
        <label>Password</label>
        <div class="input-wrapper">
        <img src="@/assets/PW_LOGO.png" class="icon" />          
          <input type="password" placeholder="Create Password" v-model="password" />
        </div>
      </div>

      <div class="form-group">
        <label>Confirm Password</label>
        <div class="input-wrapper">
        <img src="@/assets/PW_LOGO.png" class="icon" />          
        <input type="password" placeholder="Confirm Password" v-model="confirmPassword" />        </div>
      </div>

      <button class="btn" @click="register">SIGN UP</button>
      <p class="register-link">
        <router-link to="/login">Already Have an Account</router-link>
      </p>
    </div>
  </div>
</div>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { authAPI } from '@/api/index'

const authStore = useAuthStore()
const router = useRouter()

const username = ref('')
const email = ref('')
const password = ref('')
const confirmPassword = ref('')
const errorMessage = ref('')

const register = async () => {
  if(!username.value || !email.value || !password.value || !confirmPassword.value){
    errorMessage.value = 'Fill in all fields'
    return 
  }
  if(password.value != confirmPassword.value){
    errorMessage.value = 'Passwords do not match!'
    return 
  }
  try{
    const response = await authAPI.register({
      username: username.value,
      email: email.value,
      password: password.value
    })
    authStore.login(response.data)
    router.push('/home')
  }
  catch(err){
    errorMessage.value = 'Registration failed. Please try again.'
  }
}

</script>

<style scoped>

.page {
  background-color: #f5f5f5;
  min-height: 100vh;
  position: relative;
}

.logo {
  position: absolute;
  top: 10px;
  left: 20px;
  width: 40px;
  height: 40px;
  object-fit: contain;
}

.container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 80vh;
  gap: 12px;
  padding: 20px;
}

.glass-panel {
  background: rgba(255, 255, 255, 0.25);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid rgba(255, 255, 255, 0.4);
  border-radius: 24px;
  padding: 40px 50px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
}

.title {
  font-family: 'ClimateCrisis', sans-serif;
  font-size: 3.5rem;
  font-weight: 900;
  color: #FE5D26;
  margin-bottom: 10px;
}

.error-txt{
  font-family: 'JosefinSans', sans-serif;
  font-weight: 700;
  font-size: 0.95rem;
  color: #ff0000;
}

.form-group {
  display: flex;
  flex-direction: column;
  width: 320px;
  gap: 6px;
}

.form-group label {
  font-family: 'JosefinSans', sans-serif;
  font-weight: 700;
  font-size: 0.95rem;
  color: #1a1a2e;
}

.input-wrapper {
  display: flex;
  align-items: center;
  border: 1.5px solid #ccc;
  border-radius: 8px;
  padding: 10px 14px;
  background: white;
  gap: 10px;
}

.input-wrapper input {
  font-family: 'JosefinSans', sans-serif;
  border: none;
  outline: none;
  width: 100%;
  font-size: 0.8rem;
  background: transparent;
  color: #1a1a2e;
}

.icon {
  width: 18px;
  height: 18px;
  object-fit: contain;
  opacity: 0.5;
}

.forgot {
  font-family: 'JosefinSans', sans-serif;
  text-align: right;
  font-size: 0.85rem;
  color: #888;
  cursor: pointer;
  margin-top: 4px;
}

.forgot:hover {
  color: #1a1a2e;
}

.btn {
  width: 320px;
  padding: 14px;
  background-color: #1a1a2e;
  color: #ffffff;
  border: none;
  border-radius: 50px;
  font-size: 0.85rem;
  font-weight: 700;
  letter-spacing: 2px;
  cursor: pointer;
  margin-top: 8px;
  transition: background 0.2s;
}

.btn:hover {
  background-color: #FE5D26;
}

.register-link {
  font-family: 'JosefinSans', sans-serif;
  font-size: 0.85rem;
  color: #1a1a2e;
  font-weight: 600;
}

.register-link a {
  color: #1a1a2e;
  text-decoration: none;
}

.register-link a:hover {
  text-decoration: underline;
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

.logo, .container {
  position: relative;
  z-index: 1;
}

.page {
  overflow: hidden;
}
</style>