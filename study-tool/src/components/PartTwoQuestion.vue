<script setup>
import { computed, defineEmits, defineProps, ref } from "vue";

const props = defineProps([
  "videoSrc",
  "videoPoster",
  "images",
  "initialTitle",
  "imagesTitle",
  "proceedButtonLabel",
  "codeButtonLabel",
  "confirmButtonLabel",
  "showConfirmButton",
  "hint",
  "startOnImages",
  "preselectedId",
]);

const emit = defineEmits(["proceed", "confirmSelection", "imageSelected"]);

const showVideo = ref(!(props.startOnImages ?? false));

const selectedId = ref(props.preselectedId ?? null);

const proceedButtonLabel = computed(
  () => props.proceedButtonLabel ?? "Bilder anzeigen"
);
const confirmButtonLabel = computed(
  () => props.confirmButtonLabel ?? "Auswahl bestätigen"
);

const initialTitle = computed(() => props.initialTitle ?? "Einführungsvideo");
const imagesTitle = computed(
  () => props.imagesTitle ?? "Bitte wähle ein Bild aus"
);

function handleProceed() {
  showVideo.value = false;
  for (const image of props.images) {
    console.log(image);
  }
  console.log(props.images);
  emit("proceed");
}

function isSelected(img) {
  // XOR: genau ein Element – selectedId hält die aktuelle Wahl
  return (img.id ?? props.images.indexOf(img)) === selectedId.value;
}

function select(img) {
  const id = img.id ?? props.images.indexOf(img);
  selectedId.value = id;
  emit("imageSelected", { id, item: img });
}

function confirm() {
  emit("confirmSelection", selectedId);
  showVideo.value = true;
  selectedId.value = null;
}
</script>

<template>
  <section class="mc">
    <!-- Initial video view -->
    <div v-if="showVideo" class="mc__videoWrap">
      <h2 class="mc__title">{{ initialTitle }}</h2>
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
          <span class="button-text">{{ proceedButtonLabel }}</span>
        </button>
      </div>
    </div>

    <!-- Image choice view -->
    <div v-else class="mc__choices">
      <h2 class="mc__title">{{ imagesTitle }}</h2>
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
              :src="img"
              :alt="'Auswahlbild ' + (idx + 1)"
              draggable="false"
            />
            <span class="mc__check" aria-hidden="true">✓</span>
          </button>
        </li>
      </ul>

      <div class="mc__actions">
        <button
          type="button"
          class="mc__btn"
          :disabled="selectedId == null"
          @click="confirm"
          v-if="showConfirmButton"
          title="Auswahl bestätigen"
        >
          <span class="button-text">{{ confirmButtonLabel }}</span>
        </button>
      </div>
    </div>
  </section>
</template>

<style scoped>
.mc {
  display: grid;
  gap: 1rem;
}

.mc__videoWrap {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

.mc__video {
  width: 720px;
  max-height: 1280px;
  background: #000;
  margin-bottom: 3em;
  margin-top: 2em;
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

.button-text {
  font-size: 1.4em;
}

.mc__card {
  position: relative;
  display: grid;
  place-items: center;
  width: 100%;
  border: 2px solid #e5e7eb;
  background: #fff;
  cursor: pointer;
  transition: border-color 120ms ease, box-shadow 120ms ease,
    transform 60ms ease;
  outline: none;
  padding: 0;
}

.mc__card:focus-visible {
  box-shadow: 0 0 0 3px rgba(10, 50, 146, 0.35);
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
  justify-content: center;
  flex-wrap: wrap;
}

.mc__btn {
  appearance: none;
  background-color: #1e40af;
  color: #fff;
  border-radius: 999px;
  margin-top: 2em;
  padding: 1em 3em;
  font: inherit;
  cursor: pointer;
  justify-self: center;
  align-self: center;
}

.mc__btn:hover {
  background-color: #103090;
  color: #fff;
}

.mc__btn:active {
  transform: translateY(1px);
}

.mc__btn:disabled {
  opacity: 0.6;
  background: #f6f7f9;
  color: #c7cad1;
  cursor: not-allowed;
}

.mc__btn--primary {
  background-color: #1e40af;
  color: #fff;
}
</style>
