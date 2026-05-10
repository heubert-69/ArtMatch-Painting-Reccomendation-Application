import { createRouter, createWebHistory } from 'vue-router'
import LandingView from '../views/LandingView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'landing',
      component: LandingView,
    },
    {
      path: '/login',
      name: 'login',
      component: () => import('../views/LoginView.vue'),
    },
    {
      path: '/register',
      name: 'register',
      component: () => import('../views/RegisterView.vue'),
    },
    {
      path: '/home',
      name: 'home',
      component: () => import('../views/HomeView.vue'),
    },
    {
      path: '/post',
      name: 'post',
      component: () => import('../views/PostView.vue'),
    },
    {
      path: '/artist-dashboard',
      name: 'artist-dashboard',
      component: () => import('../views/ArtistDashboardView.vue'),
    },
    {
    path: '/edit-profile',
    name: 'edit-profile',
    component: () => import('../views/EditProfile.vue'),
    }
  ],
})

export default router
