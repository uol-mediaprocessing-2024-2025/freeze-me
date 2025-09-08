<script setup>
import { reactive } from "vue";
import router from "@/router";
import axios from "axios";
import { store } from "@/store.js";

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

const genderOptions = [
  { value: "weiblich", label: "Weiblich" },
  { value: "maennlich", label: "Männlich" },
  { value: "divers", label: "Divers / nichtbinär" },
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

const competencyOptions = [
  { value: "sehr_erfahren", label: "Sehr erfahren" },
  { value: "eher_erfahren", label: "Eher erfahren" },
  { value: "weniger_erfahren", label: "Weniger erfahren" },
  { value: "gar_nicht", label: "Gar nicht erfahren" },
  { value: "na", label: "Keine Angabe" },
];

const form = reactive({
  age: "",
  gender: "",
  education: "",
  frequency: "",
  competency: "",
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
    const gender = "&gender=" + normalizeEmpty(form.gender);
    const education = "&education=" + normalizeEmpty(form.education);
    const frequency = "&frequency=" + normalizeEmpty(form.frequency);
    const competency = "&competency=" + normalizeEmpty(form.competency);
    const answer = await axios.get(
      `${store.apiUrl}/send_demographic?` +
        user_path +
        age +
        gender +
        education +
        frequency +
        competency,
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
  form.alter = "";
  form.geschlecht = "";
  form.bildungsstand = "";
  form.nutzungshaeufigkeit = "";
  form.kompetenz = "";
}
</script>

<template>
  <main>
    <h1>Demographische Fragen</h1>
    <p>Im Folgendem stellen wir Ihnen ein paar Fragen zu Ihrer Person.</p>
    <form
      class="dfb"
      @submit.prevent="continue_to_part_one"
      @reset="handleReset"
      novalidate
    >
      <!-- Alter -->
      <div class="dfb_field">
        <label for="age">Alter</label>
        <select id="age" v-model="form.age" :aria-describedby="'age-help'">
          <option value="">— Bitte wählen —</option>
          <option v-for="opt in ageOptions" :key="opt.value" :value="opt.value">
            {{ opt.label }}
          </option>
        </select>
      </div>

      <!-- Geschlecht -->
      <div class="dfb_field">
        <label for="gender">Geschlecht</label>
        <select id="gender" v-model="form.gender">
          <option value="">— Bitte wählen —</option>
          <option
            v-for="opt in genderOptions"
            :key="opt.value"
            :value="opt.value"
          >
            {{ opt.label }}
          </option>
        </select>
      </div>

      <!-- Bildungsstand -->
      <div class="dfb_field">
        <label for="education">Bildungsstand</label>
        <select id="education" v-model="form.education">
          <option value="">— Bitte wählen —</option>
          <option
            v-for="opt in educationOptions"
            :key="opt.value"
            :value="opt.value"
          >
            {{ opt.label }}
          </option>
        </select>
      </div>

      <!-- Nutzungshäufigkeit -->
      <div class="dfb_field">
        <label for="frequency">Nutzungshäufigkeit digitaler Geräte</label>
        <select id="frequency" v-model="form.frequency">
          <option value="">— Bitte wählen —</option>
          <option
            v-for="opt in frequencyOptions"
            :key="opt.value"
            :value="opt.value"
          >
            {{ opt.label }}
          </option>
        </select>
      </div>

      <!-- Kompetenzeinschätzung -->
      <div class="dfb_field">
        <label for="competency"
          >Kompetenzeinschätzung (digitale Geräte/Software)</label
        >
        <select id="competency" v-model="form.competency">
          <option value="">— Bitte wählen —</option>
          <option
            v-for="opt in competencyOptions"
            :key="opt.value"
            :value="opt.value"
          >
            {{ opt.label }}
          </option>
        </select>
      </div>

      <p>
        Vielen Dank für das Beantworten der Fragen. Wenn Sie bereit sind können
        Sie mit einen Klick auf den Weiter-Knopf zum ersten Teil der Studie
        gelangen.
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

.dfb_header h2 {
  margin: 0 0 0.25rem 0;
  font-size: 1.125rem;
}

.dfb_header p {
  margin: 0 0 1rem 0;
  color: var(--muted);
  font-size: 0.9375rem;
}

.dfb_field {
  display: grid;
  gap: 0.375rem;
  margin-bottom: var(--gap);
}

label {
  font-weight: 600;
}

select {
  padding: 0.6rem 0.75rem;
  border-radius: 10px;
  border: var(--border);
  background: #fff;
  font-size: 0.98rem;
}

.dfb_actions {
  display: flex;
  flex-direction: row;
  justify-content: space-between;
  align-items: center;
  margin-top: 0.5rem;
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
}

.dfb_btn-ghost {
  background: transparent;
}
</style>
