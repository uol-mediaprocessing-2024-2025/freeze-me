<script setup>
import { reactive } from "vue";
import router from "@/router";
import store from "@/store.js";
import api from "@/api";

const ageOptions = [
  { value: "u18", label: "Unter 18" },
  { value: "18-24", label: "18–24" },
  { value: "25-34", label: "25–34" },
  { value: "35-44", label: "35–44" },
  { value: "45-54", label: "45–54" },
  { value: "55-64", label: "55–64" },
  { value: "65+", label: "65 oder älter" },
  { value: "na", label: "Keine Angabe" },
];

const colorBlindOptions = [
  { value: "color", label: "Ja" },
  { value: "nothing", label: "Nein" },
  { value: "na", label: "Keine Angabe" },
];

const educationOptions = [
  { value: "kein", label: "Kein Abschluss" },
  {
    value: "schule",
    label: "Schulabschluss (z. B. Haupt-/Real-/Mittelschule)",
  },
  { value: "abitur", label: "(Fach-)Abitur" },
  { value: "ausbildung", label: "Berufsausbildung" },
  {
    value: "hochschule",
    label: "Hochschulabschluss (Bachelor/Master/Diplom etc.)",
  },
  { value: "promotion", label: "Promotion / Habilitation" },
  { value: "na", label: "Keine Angabe" },
];

const frequencyOptions = [
  { value: "taeglich", label: "Täglich" },
  { value: "mehrmals_woche", label: "Mehrmals pro Woche" },
  { value: "seltener", label: "Seltener" },
  { value: "nie", label: "Nie" },
  { value: "na", label: "Keine Angabe" },
];

const form = reactive({
  age: "",
  colorBlind: "",
  education: "",
  frequency: "",
});

function normalizeEmpty(v) {
  if (!v || v === "na") return "na";
  return v;
}

const continue_to_part_one = async () => {
  try {
    const user_path = "user_path=" + store.user_path;
    console.log(user_path);
    const age = "&age=" + normalizeEmpty(form.age);
    const colorBlind = "&color_blind=" + normalizeEmpty(form.colorBlind);
    const education = "&education=" + normalizeEmpty(form.education);
    const frequency = "&frequency=" + normalizeEmpty(form.frequency);
    const answer = await api.get(
      `${store.apiUrl}/send_demographic?` +
        user_path +
        age +
        colorBlind +
        education +
        frequency,
      {
        responseType: "json",
      }
    );
    console.log(answer);
    if (answer.status === 200) {
      router.replace("/one");
    }
  } catch (e) {
    console.error(e);
  }
};

function handleReset() {
  form.age = "";
  form.colorBlind = "";
  form.education = "";
  form.frequency = "";
}
</script>

<template>
  <main>
    <h1>Demographische Fragen</h1>
    <p>
      Bitte beantworten Sie im Folgenden einige allgemeine Fragen. Die Angaben
      sind anonym und dienen ausschließlich dazu, die Ergebnisse der Studie
      besser einzuordnen.
    </p>
    <form
      class="dfb"
      @submit.prevent="continue_to_part_one"
      @reset="handleReset"
      novalidate
    >
      <!-- Alter -->
      <div class="dfb_field">
        <span class="dfb_label">Wie alt sind Sie?</span>
        <div class="radio-group">
          <label v-for="opt in ageOptions" :key="opt.value">
            <input type="radio" :value="opt.value" v-model="form.age" />
            <span class="radio-label">{{ opt.label }}</span>
          </label>
        </div>
      </div>

      <!-- Bildungsstand -->
      <div class="dfb_field">
        <span class="dfb_label">Was ist Ihr höchster Abschluss?</span>
        <div class="radio-group">
          <label v-for="opt in educationOptions" :key="opt.value">
            <input type="radio" :value="opt.value" v-model="form.education" />
            <span class="radio-label">{{ opt.label }}</span>
          </label>
        </div>
      </div>

      <!-- Nutzungshäufigkeit -->
      <div class="dfb_field">
        <span class="dfb_label">
          Wie häufig schauen Sie Videos auf Plattformen wie YouTube, TikTok oder
          Streamingdiensten?
        </span>
        <div class="radio-group">
          <label v-for="opt in frequencyOptions" :key="opt.value">
            <input type="radio" :value="opt.value" v-model="form.frequency" />
            <span class="radio-label">{{ opt.label }}</span>
          </label>
        </div>
      </div>

      <!-- Farbblindheit -->
      <div class="dfb_field">
        <span class="dfb_label"
          >Haben Sie eine Farbschwäche oder -blindheit?</span
        >
        <div class="radio-group">
          <label v-for="opt in colorBlindOptions" :key="opt.value">
            <input type="radio" :value="opt.value" v-model="form.colorBlind" />
            <span class="radio-label">{{ opt.label }}</span>
          </label>
        </div>
      </div>

      <p>Vielen Dank für das Beantworten der Fragen.</p>
      <p>
        Wenn Sie bereit sind können Sie mit einen Klick auf den Weiter-Knopf zum
        ersten Teil der Studie gelangen.
      </p>

      <footer class="dfb_actions">
        <button type="reset" class="dfb_btn dfb_btn-ghost">Zurücksetzen</button>
        <button type="submit" class="continue-button">WEITER</button>
      </footer>
    </form>
  </main>
</template>

<style scoped>
main {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  max-width: 700px;
  justify-self: center;
  align-self: center;
}

.continue-button {
  padding: 1em 2em;
  background-color: #1e40af;
  color: #fff;
  border-radius: 0.5em;
  margin: 1em;
  font-size: 1.4em;
  cursor: pointer;
}

.continue-button:hover {
  background-color: #0e207f;
}

h1 {
  margin: 1em;
}

.dfb {
  --gap: 1rem;
  --radius: 12px;
  --border: 1px solid rgba(0, 0, 0, 0.12);
  --bg: #fff;
  --muted: #6b7280;
  --text: #111827;

  background: var(--bg);
  color: var(--text);
  padding: 1.25rem;
  border: var(--border);
  border-radius: var(--radius);
  max-width: 680px;
  margin-top: 1em;
}

.dfb_field {
  display: grid;
  gap: 0.5rem;
  margin-bottom: 1em;
}

.dfb_field:last-of-type {
  margin-bottom: 2em;
}

.dfb_label {
  font-weight: 600;
  margin-bottom: 0.25rem;
  font-size: 1.4em;
}

.radio-group {
  display: grid;
  gap: 0.25rem;
}

.radio-group .radio-label {
  font-size: 1.3em;
}

.radio-group label {
  display: flex;
  align-items: center;
  gap: 1em;
  cursor: pointer;
  font-size: 1em;
}

.radio-group input {
  font-size: 2em;
}

.dfb_actions {
  display: flex;
  flex-direction: row;
  justify-content: space-between;
  align-items: center;
}

.dfb_btn {
  padding: 0.5em 1em;
  border: var(--border);
  color: #000;
  border-radius: 0.5em;
  margin: 1em;
  font-size: 1em;
  cursor: pointer;
  height: 4.5em;
}

.dfb_btn:hover {
  background-color: #0e207f;
  color: #fff;
}

.dfb_btn-ghost {
  background: transparent;
}
</style>
