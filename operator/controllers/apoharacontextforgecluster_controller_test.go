package controllers

import (
	"context"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	contextforgev1alpha1 "github.com/SuarezPM/Apohara_Context_Forge/operator/api/v1alpha1"
)

// newScheme returns a runtime.Scheme with all required types registered.
func newScheme(t *testing.T) *runtime.Scheme {
	t.Helper()
	s := runtime.NewScheme()
	if err := contextforgev1alpha1.AddToScheme(s); err != nil {
		t.Fatalf("AddToScheme (contextforgev1alpha1): %v", err)
	}
	if err := appsv1.AddToScheme(s); err != nil {
		t.Fatalf("AddToScheme (appsv1): %v", err)
	}
	if err := corev1.AddToScheme(s); err != nil {
		t.Fatalf("AddToScheme (corev1): %v", err)
	}
	return s
}

// sampleCluster returns a minimal ApohraContextForgeCluster CR for use in tests.
func sampleCluster(name, ns string, workerCount int32, redisURL string) *contextforgev1alpha1.ApohraContextForgeCluster {
	return &contextforgev1alpha1.ApohraContextForgeCluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: ns,
			UID:       "test-uid-1234",
		},
		Spec: contextforgev1alpha1.ApohraContextForgeClusterSpec{
			WorkerCount:     workerCount,
			Model:           "meta-llama/Llama-3-8b",
			LMCacheRedisUrl: redisURL,
			GpuType:         "mi300x",
			Image:           "ghcr.io/suarezpm/apohara-contextforge:latest",
		},
	}
}

// reconcilerFor builds a reconciler backed by a fake client pre-seeded with objs.
func reconcilerFor(t *testing.T, objs ...client.Object) *ApohraContextForgeClusterReconciler {
	t.Helper()
	s := newScheme(t)
	fakeClient := fake.NewClientBuilder().
		WithScheme(s).
		WithObjects(objs...).
		WithStatusSubresource(&contextforgev1alpha1.ApohraContextForgeCluster{}).
		Build()
	return &ApohraContextForgeClusterReconciler{
		Client: fakeClient,
		Scheme: s,
	}
}

// requestFor builds a ctrl.Request for the given name/namespace.
func requestFor(name, ns string) ctrl.Request {
	return ctrl.Request{NamespacedName: types.NamespacedName{Name: name, Namespace: ns}}
}

// ---------------------------------------------------------------------------
// Test 1 — Reconcile creates a worker Deployment when workerCount=3
// ---------------------------------------------------------------------------

func TestReconcile_CreatesWorkerDeployment(t *testing.T) {
	const (
		clusterName = "test-cluster"
		ns          = "default"
		workerCount = 3
	)

	cluster := sampleCluster(clusterName, ns, workerCount, "redis://external:6379")
	r := reconcilerFor(t, cluster)

	_, err := r.Reconcile(context.Background(), requestFor(clusterName, ns))
	if err != nil {
		t.Fatalf("Reconcile returned error: %v", err)
	}

	// The worker Deployment should exist.
	dep := &appsv1.Deployment{}
	key := types.NamespacedName{Name: clusterName + "-workers", Namespace: ns}
	if err := r.Get(context.Background(), key, dep); err != nil {
		t.Fatalf("worker Deployment not found after reconcile: %v", err)
	}

	if dep.Spec.Replicas == nil {
		t.Fatal("worker Deployment has nil replicas")
	}
	if got := *dep.Spec.Replicas; got != workerCount {
		t.Errorf("want replicas=%d, got %d", workerCount, got)
	}

	// Verify the model env var is threaded through.
	if len(dep.Spec.Template.Spec.Containers) == 0 {
		t.Fatal("worker Deployment has no containers")
	}
	container := dep.Spec.Template.Spec.Containers[0]
	foundModel := false
	for _, e := range container.Env {
		if e.Name == "MODEL" && e.Value == cluster.Spec.Model {
			foundModel = true
		}
	}
	if !foundModel {
		t.Errorf("MODEL env var not set on worker container; envs=%v", container.Env)
	}
}

// ---------------------------------------------------------------------------
// Test 2 — Reconcile provisions Redis when LMCacheRedisUrl is empty
// ---------------------------------------------------------------------------

func TestReconcile_CreatesRedisWhenNoURL(t *testing.T) {
	const (
		clusterName = "auto-redis-cluster"
		ns          = "default"
	)

	// Empty LMCacheRedisUrl triggers auto Redis provisioning.
	cluster := sampleCluster(clusterName, ns, 2, "")
	r := reconcilerFor(t, cluster)

	_, err := r.Reconcile(context.Background(), requestFor(clusterName, ns))
	if err != nil {
		t.Fatalf("Reconcile returned error: %v", err)
	}

	// Redis Deployment must exist.
	redisDep := &appsv1.Deployment{}
	redisKey := types.NamespacedName{Name: clusterName + "-redis", Namespace: ns}
	if err := r.Get(context.Background(), redisKey, redisDep); err != nil {
		t.Fatalf("Redis Deployment not found after reconcile: %v", err)
	}
	if len(redisDep.Spec.Template.Spec.Containers) == 0 {
		t.Fatal("Redis Deployment has no containers")
	}
	if redisDep.Spec.Template.Spec.Containers[0].Image != defaultRedisImage {
		t.Errorf("want Redis image=%s, got %s",
			defaultRedisImage,
			redisDep.Spec.Template.Spec.Containers[0].Image,
		)
	}

	// Redis Service must exist.
	redisSvc := &corev1.Service{}
	if err := r.Get(context.Background(), redisKey, redisSvc); err != nil {
		t.Fatalf("Redis Service not found after reconcile: %v", err)
	}
}

// ---------------------------------------------------------------------------
// Test 3 — Reconcile skips Redis when LMCacheRedisUrl is provided
// ---------------------------------------------------------------------------

func TestReconcile_SkipsRedisWhenURLProvided(t *testing.T) {
	const (
		clusterName = "byo-redis-cluster"
		ns          = "default"
		externalURL = "redis://my-external-redis.infra:6379"
	)

	cluster := sampleCluster(clusterName, ns, 2, externalURL)
	r := reconcilerFor(t, cluster)

	_, err := r.Reconcile(context.Background(), requestFor(clusterName, ns))
	if err != nil {
		t.Fatalf("Reconcile returned error: %v", err)
	}

	// Redis Deployment must NOT exist.
	redisDep := &appsv1.Deployment{}
	redisKey := types.NamespacedName{Name: clusterName + "-redis", Namespace: ns}
	err = r.Get(context.Background(), redisKey, redisDep)
	if err == nil {
		t.Error("Redis Deployment was created even though LMCacheRedisUrl was provided")
	}

	// Verify worker env has the user-supplied Redis URL.
	workerDep := &appsv1.Deployment{}
	workerKey := types.NamespacedName{Name: clusterName + "-workers", Namespace: ns}
	if err := r.Get(context.Background(), workerKey, workerDep); err != nil {
		t.Fatalf("worker Deployment not found: %v", err)
	}
	foundRedisURL := false
	for _, e := range workerDep.Spec.Template.Spec.Containers[0].Env {
		if e.Name == "LMCACHE_REDIS_URL" && e.Value == externalURL {
			foundRedisURL = true
		}
	}
	if !foundRedisURL {
		t.Errorf("LMCACHE_REDIS_URL not set to user-supplied URL; envs=%v",
			workerDep.Spec.Template.Spec.Containers[0].Env)
	}
}

// ---------------------------------------------------------------------------
// Test 4 — Reconcile updates Status.ReadyWorkers based on pod readiness
// ---------------------------------------------------------------------------

func TestReconcile_UpdatesStatusReadyWorkers(t *testing.T) {
	const (
		clusterName = "status-test-cluster"
		ns          = "default"
		workerCount = 3
	)

	cluster := sampleCluster(clusterName, ns, workerCount, "redis://external:6379")

	// Create 3 pods: 2 ready, 1 not ready.
	workerLabels := map[string]string{
		"app.kubernetes.io/name":     "apohara-contextforge",
		"app.kubernetes.io/instance": clusterName,
		workerLabelKey:               workerLabelValue,
	}

	readyPod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "worker-0", Namespace: ns, Labels: workerLabels},
		Status: corev1.PodStatus{
			Conditions: []corev1.PodCondition{
				{Type: corev1.PodReady, Status: corev1.ConditionTrue},
			},
		},
	}
	readyPod2 := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "worker-1", Namespace: ns, Labels: workerLabels},
		Status: corev1.PodStatus{
			Conditions: []corev1.PodCondition{
				{Type: corev1.PodReady, Status: corev1.ConditionTrue},
			},
		},
	}
	notReadyPod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "worker-2", Namespace: ns, Labels: workerLabels},
		Status: corev1.PodStatus{
			Conditions: []corev1.PodCondition{
				{Type: corev1.PodReady, Status: corev1.ConditionFalse},
			},
		},
	}

	r := reconcilerFor(t, cluster, readyPod, readyPod2, notReadyPod)

	_, err := r.Reconcile(context.Background(), requestFor(clusterName, ns))
	if err != nil {
		t.Fatalf("Reconcile returned error: %v", err)
	}

	// Fetch the updated CR and assert status.
	updated := &contextforgev1alpha1.ApohraContextForgeCluster{}
	if err := r.Get(context.Background(), requestFor(clusterName, ns).NamespacedName, updated); err != nil {
		t.Fatalf("get updated cluster: %v", err)
	}

	if updated.Status.ReadyWorkers != 2 {
		t.Errorf("want ReadyWorkers=2, got %d", updated.Status.ReadyWorkers)
	}
	if updated.Status.Phase != phaseDegraded {
		t.Errorf("want Phase=%s, got %s", phaseDegraded, updated.Status.Phase)
	}
}
