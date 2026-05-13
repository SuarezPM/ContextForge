package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// ApohraContextForgeClusterSpec defines the desired state of an ApohraContextForgeCluster.
type ApohraContextForgeClusterSpec struct {
	// WorkerCount is the number of vLLM+plugin worker pods to provision.
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=32
	WorkerCount int32 `json:"workerCount"`

	// Model is the HuggingFace model identifier to load on each worker.
	// Example: "meta-llama/Llama-3-8b"
	Model string `json:"model"`

	// LMCacheRedisUrl is the Redis URL for the shared LMCache backend.
	// If omitted, the operator provisions a Redis sidecar automatically.
	// +optional
	LMCacheRedisUrl string `json:"lmcacheRedisUrl,omitempty"`

	// GpuType selects the GPU hardware profile for scheduling.
	// +kubebuilder:validation:Enum=mi300x;h100;a100
	// +kubebuilder:default=mi300x
	// +optional
	GpuType string `json:"gpuType,omitempty"`

	// Image is the container image for the apohara-vllm-plugin workers.
	// Production deployments MUST pin to a digest (e.g., @sha256:...) instead of a mutable tag.
	// +kubebuilder:default="ghcr.io/suarezpm/apohara-contextforge:v7.0.0-alpha.3"
	// +optional
	Image string `json:"image,omitempty"`
}

// ApohraContextForgeClusterStatus defines the observed state of an ApohraContextForgeCluster.
type ApohraContextForgeClusterStatus struct {
	// ReadyWorkers is the count of worker pods currently in Ready state.
	ReadyWorkers int32 `json:"readyWorkers,omitempty"`

	// Phase summarises the overall cluster lifecycle state. Values emitted by
	// computePhase() in controllers/: Pending (0 ready), Degraded (1..n-1 ready),
	// Ready (all n ready). Provisioning/Failed are reserved for future use and
	// are not emitted by the current reconciler — they are NOT in the validated
	// enum to honour the V6.1 honesty discipline (claims must match runtime).
	// +kubebuilder:validation:Enum=Pending;Degraded;Ready
	Phase string `json:"phase,omitempty"`

	// RedisSecretName is the name of the auto-provisioned Redis password Secret,
	// or empty if the user supplied their own LMCacheRedisUrl.
	// +optional
	RedisSecretName string `json:"redisSecretName,omitempty"`

	// Conditions holds standard Kubernetes condition objects.
	// +optional
	// +patchMergeKey=type
	// +patchStrategy=merge
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:scope=Namespaced,shortName=acfc
// +kubebuilder:printcolumn:name="Workers",type=integer,JSONPath=".spec.workerCount"
// +kubebuilder:printcolumn:name="Ready",type=integer,JSONPath=".status.readyWorkers"
// +kubebuilder:printcolumn:name="Phase",type=string,JSONPath=".status.phase"
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=".metadata.creationTimestamp"

// ApohraContextForgeCluster is the Schema for the apoharacontextforgeclusters API.
// It represents a desired N-worker ContextForge cluster with a shared LMCache Redis backend.
type ApohraContextForgeCluster struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   ApohraContextForgeClusterSpec   `json:"spec,omitempty"`
	Status ApohraContextForgeClusterStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// ApohraContextForgeClusterList contains a list of ApohraContextForgeCluster.
type ApohraContextForgeClusterList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []ApohraContextForgeCluster `json:"items"`
}
