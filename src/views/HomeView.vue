<template>
	<div class="page">
		<div class="hero">
			<video class="bg-video" autoplay muted loop playsinline>
				<source src="@/assets/BG.mp4" type="video/mp4" />
			</video>

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
								<p class="dropdown-name">
									{{ authStore.user?.username || 'Name' }}
								</p>
								<p class="dropdown-role">
									{{ authStore.role || 'Viewer' }}
								</p>
							</div>
						</div>

						<hr />
						<p class="dropdown-item" @click="router.push('/artist-dashboard')">
							Profile
						</p>
						<p class="dropdown-item" @click="router.push('/edit-profile')">
							Settings
						</p>
						<p class="dropdown-item" @click="router.push('/')">
							Sign Out
						</p>
					</div>
				</div>
			</nav>

			<div class="header">
				<h1 class="brand">ArtMatch</h1>

				<div class="search-wrapper">
					<input
						type="text"
						placeholder="Paste Google Drive Image Link..."
						v-model="driveLink"
						class="search-input"
					/>
					<span class="search-icon">🔍</span>
				</div>

				<button class="fab" @click="getRecommendations">+</button>

				<p class="section-title">Recommended for you</p>
			</div>
		</div>

		<div class="grid">
			<div
				class="card"
				v-for="(painting, index) in paintings"
				:key="index"
				@click="getRecommendations(painting.file_id)"
			>
				<img class="card-image" :src="painting.image_url" />

				<div class="card-info">
					<p class="card-title">{{ painting.file_id }}</p>
					<p class="card-artist">
						Score: {{ painting.score?.toFixed(3) }}
					</p>
				</div>
			</div>
		</div>
	</div>
</template>

<script setup>
import { useRouter } from 'vue-router'
import { ref, onMounted, onUnmounted } from 'vue'
import { useAuthStore } from '@/stores/auth'
import { paintingsAPI } from '@/api'

const router = useRouter()
const authStore = useAuthStore()

const scrolled = ref(false)
const showDropdown = ref(false)

const driveLink = ref("")
const paintings = ref([])

const extractFileId = (input) => {
	if (!input) return null

	if (!input.includes("/")) return input

	let match = input.match(/\/d\/([^\/]+)/)
	if (match) return match[1]

	match = input.match(/[?&]id=([^&]+)/)
	if (match) return match[1]

	return null
}

const handleScroll = () => {
	scrolled.value = window.scrollY > 50
}

onMounted(() => {
	window.addEventListener('scroll', handleScroll)

	getRecommendations("1cVaTHvzg5Tyl9wLlLtUThkUgG74yAmZE")

	console.log("DRIVE INPUT:", driveLink.value)
	console.log("EXTRACTED:", extractFileId(driveLink.value))
})

onUnmounted(() => {
	window.removeEventListener('scroll', handleScroll)
})

const toggleDropdown = () => {
	showDropdown.value = !showDropdown.value
}

const getRecommendations = async (id) => {
	try {
		const file =
			id ||
			extractFileId(driveLink.value)

		if (!file) return

		const res = await paintingsAPI.recommend({
			file_id: file,
			top_k: 12
		})

		paintings.value = res.data.recommendations
	} catch (err) {
		console.error("API error:", err)
	}
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
	width: 100%;
	height: 100%;
	object-fit: cover;
}

.navbar {
	display: flex;
	justify-content: space-between;
	padding: 12px 24px;
	position: sticky;
	top: 0;
	background: white;
	z-index: 10;
}

.logo {
	width: 40px;
}

.profile-icon {
	width: 36px;
	border-radius: 50%;
	cursor: pointer;
}

.header {
	display: flex;
	flex-direction: column;
	align-items: center;
}

.brand {
	font-size: 7rem;
	color: #FE5D26;
}

.search-wrapper {
	display: flex;
	border: 1px solid #ccc;
	border-radius: 50px;
	padding: 8px;
	width: 400px;
}

.search-input {
	border: none;
	outline: none;
	width: 100%;
}

.fab {
	position: fixed;
	bottom: 30px;
	left: 50%;
	transform: translateX(-50%);
	background: #FE5D26;
	color: white;
	border-radius: 50%;
	width: 56px;
	height: 56px;
	border: none;
	font-size: 24px;
}

.grid {
	display: grid;
	grid-template-columns: repeat(4, 1fr);
	gap: 8px;
	padding: 24px;
}

.card {
	cursor: pointer;
}

.card-image {
	width: 100%;
	height: 160px;
	object-fit: cover;
	background: #ddd;
}

.card-title {
	font-size: 0.85rem;
}

.card-artist {
	font-size: 0.75rem;
}
</style>