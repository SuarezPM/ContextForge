{{/*
Expand the name of the chart.
*/}}
{{- define "apohara-contextforge.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
Truncates at 63 chars because some Kubernetes name fields have this limit.
*/}}
{{- define "apohara-contextforge.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart label (name + version).
*/}}
{{- define "apohara-contextforge.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels applied to all resources.
*/}}
{{- define "apohara-contextforge.labels" -}}
helm.sh/chart: {{ include "apohara-contextforge.chart" . }}
{{ include "apohara-contextforge.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels used in matchLabels.
*/}}
{{- define "apohara-contextforge.selectorLabels" -}}
app.kubernetes.io/name: {{ include "apohara-contextforge.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Redis service URL: prefer user-supplied; fall back to in-cluster Redis pod.
*/}}
{{- define "apohara-contextforge.redisUrl" -}}
{{- if .Values.lmcacheRedisUrl }}
{{- .Values.lmcacheRedisUrl }}
{{- else }}
{{- printf "redis://%s-redis:%d" (include "apohara-contextforge.fullname" .) (.Values.redis.port | int) }}
{{- end }}
{{- end }}
