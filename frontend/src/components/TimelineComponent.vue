<script setup>
import {onMounted, ref, watch} from "vue";
import {store} from "@/store.js";
import axios from "axios";

const segmentation = ref(store.steps.segmentation);
const main_effect = ref(store.steps.mainEffect);
const after_effect = ref(store.steps.afterEffect);
const video_id = ref(store.selectedVideoId)

const props = defineProps(['modelValue'])
const emit = defineEmits(['update:modelValue'])
const goToPage = (index) => {
  emit('update:modelValue', index)
}

onMounted(async () => {
  if (store.selectedVideoId != null) {
    video_id.value = store.selectedVideoId
    const video_data = (await axios.get(`${store.apiUrl}/project-data?video_id=` + store.selectedVideoId)).data
    console.log(video_data)
    store.steps.segmentation = video_data.available_steps.includes("segmentation")
    store.steps.mainEffect = video_data.available_steps.includes("main-effect")
    store.steps.afterEffect = video_data.available_steps.includes("after-effect")
    console.log(store)
  }
})

watch(
  () => store.selectedVideoId,
  async (newVideoId) => {
    if (newVideoId == null) {
      segmentation.value = false;
      main_effect.value = false;
      after_effect.value = false;
      return;
    }
    video_id.value = newVideoId
  }
)

const goToStep = (step) => {
  switch (step) {
    case 'segmentation':
      if (!store.steps.segmentation) {
        return
      }
      goToPage(2)
      break;
    case 'main-effect':
      if (!store.steps.mainEffect) {
        return
      }
      goToPage(3)
      break;
    case 'after-effect':
      if (!store.steps.afterEffect) {
        return
      }
      goToPage(4)
      break;
    default:
      goToPage(1)
  }
}

</script>

<template>
  <v-card v-if="video_id" class="progress-line" elevation="4">
    <v-card class="step-card clickable" elevation="3" @click="goToStep('video-editing')">
      <img :src="'src/assets/workflow/video-cut-available.svg'"
         class="progress-icon" :class="props.modelValue === 1 ? 'current-step' : '' ">
      <p>Video Editing</p>
    </v-card>

    <span class="divider" :class="segmentation ? 'colored-divider' : ''"></span>
    <v-card class="step-card" :class="segmentation ? 'clickable' : ''" elevation="3" @click="goToStep('segmentation')" :disabled="!segmentation">
      <img :src="segmentation ? 'src/assets/workflow/segmentation-available.svg' : 'src/assets/workflow/segmentation-unavailable.svg'"
         class="progress-icon" :class="props.modelValue === 2 ? 'current-step' : '' ">
      <p>Segmentation</p>
    </v-card>

    <span class="divider" :class="main_effect ? 'colored-divider' : ''"></span>
    <v-card class="step-card" :class="main_effect ? 'clickable' : ''" elevation="3" @click="goToStep('main-effect')" :disabled="!main_effect">
      <img :src="main_effect ? 'src/assets/workflow/main-effect-available.svg' : 'src/assets/workflow/main-effect-unavailable.svg'"
         class="progress-icon" :class="props.modelValue === 3 ? 'current-step' : '' ">
      <p>Main-Effect</p>
    </v-card>

    <span class="divider" :class="after_effect ? 'colored-divider' : ''"></span>
    <v-card class="step-card" :class="after_effect ? 'clickable' : ''" elevation="3" @click="goToStep('after-effect')" :disabled="!after_effect">
      <img :src="after_effect ? 'src/assets/workflow/after-effect-available.svg' : 'src/assets/workflow/after-effect-unavailable.svg'"
         class="progress-icon" :class="props.modelValue === 4 ? 'current-step' : '' ">
      <p>After-Effects</p>
    </v-card>
  </v-card>
</template>

<style scoped>
.progress-line {
  display: flex;
  flex-direction: column;
  flex-wrap: nowrap;
  align-items: center;
  padding: 1em 0.5em;
  width: 10em;
  min-width: 8em;
  margin-right: 1em;
}

.step-card {
  justify-items: center;
  width: 7em;
  padding: 0.5em 0;
}

.step-card:hover {
  cursor: default;
  background: rgba(0,0,0,0);
}

.clickable:hover {
  background: rgba(0,0,0,0.1);
  cursor: pointer;
}

.progress-icon {
  height: 2em;
  width: 2em;
}

.divider {
  height: 2em;
  border-left: 2px solid dimgray;
  margin: 5px
}

.colored-divider {
  border-color: blue;
}

.current-step {
  border: blue 1px solid;
  border-radius: 0.3em;
}
</style>