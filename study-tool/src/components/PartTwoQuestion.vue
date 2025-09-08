<!-- MediaChoice.vue -->
<template>
  <section class="mc">
    <!-- Headline above video or images -->
    <h2 class="mc__title" v-if="showVideo">{{ initialTitle }}</h2>
    <h2 class="mc__title" v-else>{{ imagesTitle }}</h2>

    <!-- Initial video view -->
    <div v-if="showVideo" class="mc__videoWrap">
      <video
        class="mc__video"
        :src="videoSrc"
        :poster="videoPoster"
        controls
        playsinline
      />
      <div class="mc__actions">
        <button
          type="button"
          class="mc__btn mc__btn--primary"
          @click="handleProceed"
        >
          {{ proceedButtonLabel }}
        </button>

        <button type="button" class="mc__btn" @click="$emit('codeClicked')">
          {{ codeButtonLabel }}
        </button>
      </div>
    </div>

    <!-- Image choice view -->
    <div v-else class="mc__choices">
      <p class="mc__hint" v-if="hint">{{ hint }}</p>

      <ul class="mc__grid" role="list">
        <li v-for="(img, idx) in images" :key="img.id ?? idx">
          <button
            type="button"
            class="mc__card"
            :class="{ 'is-selected': isSelected(img) }"
            :aria-pressed="isSelected(img) ? 'true' : 'false'"
            @click="select(img)"
            @keydown.enter.prevent="select(img)"
            @keydown.space.prevent="select(img)"
          >
            <img
              class="mc__img"
              :src="img.src"
              :alt="img.alt || 'Auswahlbild ' + (idx + 1)"
              draggable="false"
            />
            <span class="mc__check" aria-hidden="true">✓</span>
          </button>
        </li>
      </ul>

      <div class="mc__actions">
        <button type="button" class="mc__btn" @click="$emit('codeClicked')">
          {{ codeButtonLabel }}
        </button>

        <button
          type="button"
          class="mc__btn"
          :disabled="!selectedId"
          @click="$emit('confirmSelection', selectedId)"
          v-if="showConfirmButton"
          title="Auswahl bestätigen"
        >
          {{ confirmButtonLabel }}
        </button>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, ref } from "vue";

type ImageItem = {
  id?: string | number;
  src: string;
  alt?: string;
};

const props = defineProps<{
  videoSrc: string;
  videoPoster?: string;
  images: ImageItem[]; // Erwartet 3 Einträge
  initialTitle?: string;
  imagesTitle?: string;
  proceedButtonLabel?: string;
  codeButtonLabel?: string;
  confirmButtonLabel?: string;
  showConfirmButton?: boolean;
  hint?: string;
  startOnImages?: boolean;
  preselectedId?: string | number | null;
}>();

const emit = defineEmits<{
  (e: "proceed"): void;
  (
    e: "imageSelected",
    payload: { id: string | number | undefined; item: ImageItem }
  ): void;
  (e: "confirmSelection", id: string | number | undefined): void;
  (e: "codeClicked"): void;
}>();

const showVideo = ref(!(props.startOnImages ?? false));

const selectedId = ref<string | number | undefined | null>(
  props.preselectedId ?? null
);

const proceedButtonLabel = computed(
  () => props.proceedButtonLabel ?? "Bilder anzeigen"
);
const codeButtonLabel = computed(() => props.codeButtonLabel ?? "Code");
const confirmButtonLabel = computed(
  () => props.confirmButtonLabel ?? "Auswahl bestätigen"
);

const initialTitle = computed(() => props.initialTitle ?? "Einführungsvideo");
const imagesTitle = computed(
  () => props.imagesTitle ?? "Bitte wähle ein Bild aus"
);

function handleProceed() {
  showVideo.value = false;
  emit("proceed");
}

function isSelected(img: ImageItem) {
  // XOR: genau ein Element – selectedId hält die aktuelle Wahl
  return (img.id ?? props.images.indexOf(img)) === selectedId.value;
}

function select(img: ImageItem) {
  const id = img.id ?? props.images.indexOf(img);
  selectedId.value = id;
  emit("imageSelected", { id, item: img });
}
</script>

<style scoped>
.mc {
  display: grid;
  gap: 1rem;
}

.mc__title {
  margin: 0;
  font-size: 1.25rem;
  line-height: 1.2;
}

.mc__videoWrap {
  display: grid;
  gap: 0.75rem;
}

.mc__video {
  width: 100%;
  max-height: 60vh;
  border-radius: 12px;
  background: #000;
}

.mc__hint {
  margin: 0 0 0.25rem 0;
  font-size: 0.95rem;
  color: #555;
}

.mc__grid {
  list-style: none;
  padding: 0;
  margin: 0;
  display: grid;
  gap: 0.75rem;
  grid-template-columns: repeat(3, minmax(0, 1fr));
}

.mc__card {
  position: relative;
  display: grid;
  place-items: center;
  width: 100%;
  aspect-ratio: 4 / 3;
  border: 2px solid #e5e7eb;
  border-radius: 12px;
  background: #fff;
  cursor: pointer;
  transition: border-color 120ms ease, box-shadow 120ms ease,
    transform 60ms ease;
  outline: none;
}

.mc__card:focus-visible {
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.35);
}

.mc__card:hover {
  border-color: #c7cad1;
}

.mc__card.is-selected {
  border-color: #2563eb;
  box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.2);
}

.mc__img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  border-radius: 10px;
  user-select: none;
  pointer-events: none;
}

.mc__check {
  position: absolute;
  top: 8px;
  right: 10px;
  font-weight: 700;
  opacity: 0;
  transform: scale(0.9);
  transition: opacity 120ms ease, transform 120ms ease;
}

.mc__card.is-selected .mc__check {
  opacity: 1;
  transform: scale(1);
}

.mc__actions {
  display: flex;
  gap: 0.5rem;
  justify-content: flex-start;
  flex-wrap: wrap;
}

.mc__btn {
  appearance: none;
  border: 1px solid #d1d5db;
  background: #fff;
  border-radius: 999px;
  padding: 0.55rem 0.9rem;
  font: inherit;
  cursor: pointer;
  transition: background 120ms ease, border-color 120ms ease,
    transform 40ms ease;
}

.mc__btn:hover {
  background: #f6f7f9;
  border-color: #c7cad1;
}

.mc__btn:active {
  transform: translateY(1px);
}

.mc__btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.mc__btn--primary {
  border-color: #2563eb;
  background: #2563eb;
  color: white;
}
</style>
