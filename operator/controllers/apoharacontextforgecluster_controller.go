package controllers

import (
	"context"
	"fmt"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	contextforgev1alpha1 "github.com/SuarezPM/Apohara_Context_Forge/operator/api/v1alpha1"
)

const (
	// workerLabelKey is the label applied to worker pods for selection.
	workerLabelKey = "app.kubernetes.io/component"
	// workerLabelValue is the value that identifies contextforge worker pods.
	workerLabelValue = "contextforge-worker"
	// redisLabelValue is the label value applied to the managed Redis deployment.
	redisLabelValue = "contextforge-redis"
	// redisSidecarPort is the default Redis port.
	redisSidecarPort = 6379
	// defaultRedisImage is the Redis image used for the auto-provisioned sidecar.
	defaultRedisImage = "redis:7-alpine"
	// phaseReady indicates all requested workers are ready.
	phaseReady = "Ready"
	// phaseDegraded indicates fewer than the requested number of workers are ready.
	phaseDegraded = "Degraded"
	// phasePending indicates no workers are ready yet.
	phasePending = "Pending"
)

// ApohraContextForgeClusterReconciler reconciles an ApohraContextForgeCluster object.
type ApohraContextForgeClusterReconciler struct {
	client.Client
	Scheme *runtime.Scheme
}

// +kubebuilder:rbac:groups=contextforge.apohara.dev,resources=apoharacontextforgeclusters,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=contextforge.apohara.dev,resources=apoharacontextforgeclusters/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=contextforge.apohara.dev,resources=apoharacontextforgeclusters/finalizers,verbs=update
// +kubebuilder:rbac:groups=apps,resources=deployments,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=core,resources=services;configmaps;pods,verbs=get;list;watch;create;update;patch;delete

// Reconcile is the main controller loop for ApohraContextForgeCluster.
//
// It ensures:
//  1. A Redis Deployment+Service exists in the cluster namespace when
//     LMCacheRedisUrl is not provided by the user (auto-sidecar mode).
//  2. A worker Deployment exists with exactly Spec.WorkerCount replicas.
//  3. Status.ReadyWorkers and Status.Phase are kept in sync with observed state.
//
// The reconciler is level-driven and re-queues every 30 s to catch drift
// caused by external changes (e.g. manual pod deletes).
func (r *ApohraContextForgeClusterReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	// 1. Fetch the ApohraContextForgeCluster CR.
	cluster := &contextforgev1alpha1.ApohraContextForgeCluster{}
	if err := r.Get(ctx, req.NamespacedName, cluster); err != nil {
		// NotFound means the object was deleted between the queue event and now.
		// Return without error so the controller does not re-queue.
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	logger.Info("reconciling cluster",
		"name", cluster.Name,
		"namespace", cluster.Namespace,
		"workerCount", cluster.Spec.WorkerCount,
	)

	// 2. Optional Redis sidecar — only when the user has not supplied their own Redis URL.
	if cluster.Spec.LMCacheRedisUrl == "" {
		if err := r.reconcileRedisSidecar(ctx, cluster); err != nil {
			return ctrl.Result{}, fmt.Errorf("reconcileRedisSidecar: %w", err)
		}
	}

	// 3. Worker Deployment — ensure exactly Spec.WorkerCount replicas exist.
	if err := r.reconcileWorkers(ctx, cluster); err != nil {
		return ctrl.Result{}, fmt.Errorf("reconcileWorkers: %w", err)
	}

	// 4. Status — count ready pods and update phase.
	if err := r.updateStatus(ctx, cluster); err != nil {
		return ctrl.Result{}, fmt.Errorf("updateStatus: %w", err)
	}

	// Re-queue after 30 s to catch any drift not driven by events.
	return ctrl.Result{RequeueAfter: 30 * time.Second}, nil
}

// reconcileRedisSidecar ensures a Redis Deployment and its ClusterIP Service
// exist in cluster.Namespace. When both already exist, no changes are made.
// The created Service is named "<cluster.Name>-redis" and the operator sets
// it as an owner reference so it is garbage-collected when the CR is deleted.
func (r *ApohraContextForgeClusterReconciler) reconcileRedisSidecar(ctx context.Context, cluster *contextforgev1alpha1.ApohraContextForgeCluster) error {
	logger := log.FromContext(ctx)
	redisName := cluster.Name + "-redis"
	ns := cluster.Namespace

	// --- Deployment ---
	redisLabels := map[string]string{
		"app.kubernetes.io/name":       "apohara-contextforge",
		"app.kubernetes.io/instance":   cluster.Name,
		workerLabelKey:                 redisLabelValue,
	}
	redisDeployment := &appsv1.Deployment{}
	err := r.Get(ctx, types.NamespacedName{Name: redisName, Namespace: ns}, redisDeployment)
	if errors.IsNotFound(err) {
		one := int32(1)
		desired := &appsv1.Deployment{
			ObjectMeta: metav1.ObjectMeta{
				Name:      redisName,
				Namespace: ns,
				Labels:    redisLabels,
				OwnerReferences: []metav1.OwnerReference{
					ownerRef(cluster, r.Scheme),
				},
			},
			Spec: appsv1.DeploymentSpec{
				Replicas: &one,
				Selector: &metav1.LabelSelector{MatchLabels: redisLabels},
				Template: corev1.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{Labels: redisLabels},
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{
							{
								Name:  "redis",
								Image: defaultRedisImage,
								Ports: []corev1.ContainerPort{
									{ContainerPort: redisSidecarPort, Protocol: corev1.ProtocolTCP},
								},
								Resources: corev1.ResourceRequirements{},
							},
						},
					},
				},
			},
		}
		if err := r.Create(ctx, desired); err != nil {
			return fmt.Errorf("create Redis Deployment %s/%s: %w", ns, redisName, err)
		}
		logger.Info("created Redis Deployment", "name", redisName, "namespace", ns)
	} else if err != nil {
		return fmt.Errorf("get Redis Deployment %s/%s: %w", ns, redisName, err)
	}

	// --- Service ---
	redisSvc := &corev1.Service{}
	err = r.Get(ctx, types.NamespacedName{Name: redisName, Namespace: ns}, redisSvc)
	if errors.IsNotFound(err) {
		desired := &corev1.Service{
			ObjectMeta: metav1.ObjectMeta{
				Name:      redisName,
				Namespace: ns,
				Labels:    redisLabels,
				OwnerReferences: []metav1.OwnerReference{
					ownerRef(cluster, r.Scheme),
				},
			},
			Spec: corev1.ServiceSpec{
				Selector: redisLabels,
				Ports: []corev1.ServicePort{
					{Port: redisSidecarPort, Protocol: corev1.ProtocolTCP},
				},
				Type: corev1.ServiceTypeClusterIP,
			},
		}
		if err := r.Create(ctx, desired); err != nil {
			return fmt.Errorf("create Redis Service %s/%s: %w", ns, redisName, err)
		}
		logger.Info("created Redis Service", "name", redisName, "namespace", ns)
	} else if err != nil {
		return fmt.Errorf("get Redis Service %s/%s: %w", ns, redisName, err)
	}

	return nil
}

// reconcileWorkers ensures a single Deployment with cluster.Spec.WorkerCount
// replicas exists in cluster.Namespace. If the Deployment already exists its
// replica count and image are updated to match the spec.
func (r *ApohraContextForgeClusterReconciler) reconcileWorkers(ctx context.Context, cluster *contextforgev1alpha1.ApohraContextForgeCluster) error {
	logger := log.FromContext(ctx)
	workerName := cluster.Name + "-workers"
	ns := cluster.Namespace

	workerLabels := map[string]string{
		"app.kubernetes.io/name":       "apohara-contextforge",
		"app.kubernetes.io/instance":   cluster.Name,
		workerLabelKey:                 workerLabelValue,
	}

	image := cluster.Spec.Image
	if image == "" {
		image = "ghcr.io/suarezpm/apohara-contextforge:latest"
	}

	// Resolve the Redis URL: use what the user provided or the auto-provisioned sidecar.
	redisURL := cluster.Spec.LMCacheRedisUrl
	if redisURL == "" {
		redisURL = fmt.Sprintf("redis://%s-redis.%s.svc.cluster.local:%d", cluster.Name, ns, redisSidecarPort)
	}

	replicas := cluster.Spec.WorkerCount

	existing := &appsv1.Deployment{}
	err := r.Get(ctx, types.NamespacedName{Name: workerName, Namespace: ns}, existing)
	if errors.IsNotFound(err) {
		desired := r.workerDeployment(cluster, workerName, ns, workerLabels, image, redisURL, replicas)
		if err := r.Create(ctx, desired); err != nil {
			return fmt.Errorf("create worker Deployment %s/%s: %w", ns, workerName, err)
		}
		logger.Info("created worker Deployment", "name", workerName, "replicas", replicas)
		return nil
	}
	if err != nil {
		return fmt.Errorf("get worker Deployment %s/%s: %w", ns, workerName, err)
	}

	// Deployment exists — reconcile replica count and image if they drifted.
	needsUpdate := false
	if existing.Spec.Replicas == nil || *existing.Spec.Replicas != replicas {
		existing.Spec.Replicas = &replicas
		needsUpdate = true
	}
	if len(existing.Spec.Template.Spec.Containers) > 0 &&
		existing.Spec.Template.Spec.Containers[0].Image != image {
		existing.Spec.Template.Spec.Containers[0].Image = image
		needsUpdate = true
	}
	if needsUpdate {
		if err := r.Update(ctx, existing); err != nil {
			return fmt.Errorf("update worker Deployment %s/%s: %w", ns, workerName, err)
		}
		logger.Info("updated worker Deployment", "name", workerName, "replicas", replicas)
	}
	return nil
}

// workerDeployment builds the desired appsv1.Deployment object for the worker fleet.
func (r *ApohraContextForgeClusterReconciler) workerDeployment(
	cluster *contextforgev1alpha1.ApohraContextForgeCluster,
	name, ns string,
	lbls map[string]string,
	image, redisURL string,
	replicas int32,
) *appsv1.Deployment {
	return &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: ns,
			Labels:    lbls,
			OwnerReferences: []metav1.OwnerReference{
				ownerRef(cluster, r.Scheme),
			},
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{MatchLabels: lbls},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: lbls},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:  "contextforge-worker",
							Image: image,
							Env: []corev1.EnvVar{
								{
									Name:  "LMCACHE_REDIS_URL",
									Value: redisURL,
								},
								{
									Name:  "MODEL",
									Value: cluster.Spec.Model,
								},
							},
							Ports: []corev1.ContainerPort{
								{Name: "http", ContainerPort: 8000, Protocol: corev1.ProtocolTCP},
							},
							ReadinessProbe: &corev1.Probe{
								ProbeHandler: corev1.ProbeHandler{
									HTTPGet: &corev1.HTTPGetAction{
										Path: "/health",
										Port: intstr.FromInt32(8000),
									},
								},
								InitialDelaySeconds: 30,
								PeriodSeconds:       10,
								FailureThreshold:    3,
							},
						},
					},
					// GPU node affinity — schedule on nodes advertising the requested GPU type.
					NodeSelector: gpuNodeSelector(cluster.Spec.GpuType),
				},
			},
		},
	}
}

// updateStatus counts ready worker pods via label selector and sets
// cluster.Status.ReadyWorkers and cluster.Status.Phase accordingly.
func (r *ApohraContextForgeClusterReconciler) updateStatus(ctx context.Context, cluster *contextforgev1alpha1.ApohraContextForgeCluster) error {
	logger := log.FromContext(ctx)

	podList := &corev1.PodList{}
	labelSel := labels.SelectorFromSet(labels.Set{
		"app.kubernetes.io/instance": cluster.Name,
		workerLabelKey:               workerLabelValue,
	})
	if err := r.List(ctx, podList,
		client.InNamespace(cluster.Namespace),
		client.MatchingLabelsSelector{Selector: labelSel},
	); err != nil {
		return fmt.Errorf("list worker pods: %w", err)
	}

	var readyCount int32
	for i := range podList.Items {
		pod := &podList.Items[i]
		if isPodReady(pod) {
			readyCount++
		}
	}

	desiredPhase := computePhase(readyCount, cluster.Spec.WorkerCount)

	// Only patch status when it has actually changed to avoid noisy updates.
	if cluster.Status.ReadyWorkers == readyCount && cluster.Status.Phase == desiredPhase {
		return nil
	}

	patch := client.MergeFrom(cluster.DeepCopy())
	cluster.Status.ReadyWorkers = readyCount
	cluster.Status.Phase = desiredPhase

	// Update Available condition.
	availCondition := metav1.Condition{
		Type:               "Available",
		Status:             metav1.ConditionFalse,
		Reason:             "WorkersNotReady",
		Message:            fmt.Sprintf("%d/%d workers ready", readyCount, cluster.Spec.WorkerCount),
		LastTransitionTime: metav1.Now(),
	}
	if readyCount >= cluster.Spec.WorkerCount {
		availCondition.Status = metav1.ConditionTrue
		availCondition.Reason = "AllWorkersReady"
		availCondition.Message = fmt.Sprintf("All %d workers are ready", cluster.Spec.WorkerCount)
	}
	setCondition(&cluster.Status.Conditions, availCondition)

	if err := r.Status().Patch(ctx, cluster, patch); err != nil {
		return fmt.Errorf("patch status: %w", err)
	}
	logger.Info("status updated",
		"readyWorkers", readyCount,
		"phase", desiredPhase,
	)
	return nil
}

// SetupWithManager registers the reconciler with the controller-runtime manager
// and sets up watches on owned Deployments so changes bubble up to the CR.
func (r *ApohraContextForgeClusterReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&contextforgev1alpha1.ApohraContextForgeCluster{}).
		Owns(&appsv1.Deployment{}).
		Owns(&corev1.Service{}).
		Complete(r)
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

// ownerRef constructs an OwnerReference pointing at cluster so child objects
// are garbage-collected when the CR is deleted.
func ownerRef(cluster *contextforgev1alpha1.ApohraContextForgeCluster, scheme *runtime.Scheme) metav1.OwnerReference {
	gvks, _, _ := scheme.ObjectKinds(cluster)
	var apiVersion, kind string
	if len(gvks) > 0 {
		apiVersion = gvks[0].GroupVersion().String()
		kind = gvks[0].Kind
	}
	t := true
	return metav1.OwnerReference{
		APIVersion:         apiVersion,
		Kind:               kind,
		Name:               cluster.Name,
		UID:                cluster.UID,
		Controller:         &t,
		BlockOwnerDeletion: &t,
	}
}

// isPodReady returns true if the pod has passed its readiness probe
// (PodReady condition is True).
func isPodReady(pod *corev1.Pod) bool {
	for _, cond := range pod.Status.Conditions {
		if cond.Type == corev1.PodReady && cond.Status == corev1.ConditionTrue {
			return true
		}
	}
	return false
}

// computePhase derives the cluster lifecycle phase from the ready/desired counts.
func computePhase(ready, desired int32) string {
	switch {
	case ready == 0:
		return phasePending
	case ready < desired:
		return phaseDegraded
	default:
		return phaseReady
	}
}

// setCondition upserts cond into conditions by Type.
func setCondition(conditions *[]metav1.Condition, cond metav1.Condition) {
	for i, existing := range *conditions {
		if existing.Type == cond.Type {
			(*conditions)[i] = cond
			return
		}
	}
	*conditions = append(*conditions, cond)
}

// gpuNodeSelector returns a label map that constrains pods to nodes that
// expose the requested GPU type via standard AMD/NVIDIA node labels.
func gpuNodeSelector(gpuType string) map[string]string {
	if gpuType == "" {
		return nil
	}
	return map[string]string{
		"apohara.dev/gpu-type": gpuType,
	}
}

