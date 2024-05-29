clc; close all; clear;
% Code for "Mapping back and forth between model predictive control and neural
% networks"  presented at the Learning for Dynamics and Control Conference, 2024.
% Authors: Ross Drummond, Pablo Baldivieso, and Giorgio Valmorbida.

% The code generates an implicit neural network (NN) characterisation of a model
% predictive control (MPC) policy. This implicit NN is unravelled using the
% method discussed in the report.


%% MPC problem setup
n = 10; nx = 2*n;
scaling = 1*1e0;

A = [4/3, -2/3;1 0]; B =[0;1];
N = 2;

Rtilde = 1; Ptilde = [7.1667,-4.2222;-4.2222 , 4.6852]; Qtilde = [1,-2/3;-2/3 , 3/22];

R = Rtilde; P = Ptilde;
for j = 1:n-1
    R = blkdiag(R,Rtilde); P = blkdiag(Qtilde,P);
end

My = zeros(N*n,n);
Gtilde = [0.1;-0.1]; G = Gtilde;
for i = 1:n
    for j = 1:i
        My((i-1)*(N)+1:i*N,j) = (A^(i-j))*B;
    end
    if i<n
        G = blkdiag(G,Gtilde);
    end
end

H = R + My'*P*My;
F_add_quadprog = 1*[A'*Qtilde*B,zeros(N,n-1)]';
F_add = F_add_quadprog*1;
S = G*(inv(H))*F_add;

w = ones(2*n,1)*scaling;

D = (eye(nx)-G*(inv(H))*G');

opt_details.set_solver = 'sedumi';

b_qp = w;
A_qp = G;

%% Simulate the MPC policy. Details from paper. 
K = 25;

x0 = -[1;1]*2e2;
xk = x0;
u = sdpvar(n,1);
u_mpc = zeros(K,1); u_YALMIP = zeros(n,K); xk_store_MPC = zeros(N,K);

for j = 1:K
    f_yalmip =  xk'*2*[A'*Qtilde*B,zeros(N,n-1)];
    cost = u'*H*u + 1e0*f_yalmip*u;
    Constraints = [A_qp*u <= b_qp];
    optimize(Constraints,cost,sdpsettings('solver',opt_details.set_solver));
    u_YALMIP(:,j) = value(u);
    
    u_mpc(j) = u_YALMIP(1,j);
    xk = A*xk+B*u_mpc(j);
    xk_store_MPC(:,j) = xk;
end

xk_store_MPC;


%% Simulate the implicit neural network policy.
% The code unravels the implicit NN into an explicit one, as detailed in
% the pape.r 

iters = 1e3; % set the number of iterations of the NN unravlelling
y_store = zeros(nx,iters); phi_store = zeros(nx,iters); res_store = zeros(nx,iters);res_norm = zeros(iters,1); sign_store = ones(nx,iters);

xk = x0;
c_MPC = 1*S*xk+ 1*w;% initialise the implicit NN.
zeta = -1*c_MPC;
        
y0 = zeros(nx,1);
phi = zeros(nx,1);
for g = 1:nx
    if y0(g) >=0
        phi(g) = y0(g);
    else
        phi(g) = 0;
    end
end

residual = y0-D*phi-zeta;
K_gain = 0; y = y0;

u_nn = zeros(K,1); u_ramp = zeros(n,K); xk_store_nn = zeros(N,K);
res_norm_store = zeros(K,iters);
for k = 1:K
    for j = 1:iters % unravel the implicit NN into an explicit one
        c_MPC = 1*S*xk+ 1*w; %define the implicit NN weights
        zeta = -1*c_MPC; 
        y_store (:,j) = y;
        phi_store (:,j) = phi; res_store (:,j) = residual;
        ykp1 = D*phi + zeta + 1*K_gain*(residual);
        
        phi = zeros(n,1);
        for g = 1:nx
            if ykp1(g) >0
                phi(g) = ykp1(g);
            else
                phi(g) = 0;
                sign_store(g,j) = -1;
            end
        end
        y  = ykp1;
        residual = y-D*phi-zeta; % compute residuals
        res_norm(j) = norm(residual); 
        res_norm_store(k,j)= norm(y-D*phi-zeta);
    end
    
    y_ramp = ykp1;
    u_ramp(:,k) = -1*inv(H)*(F_add*xk+G'*phi); % get the output
    
    u_nn(k) = u_ramp(1,k); % the step-ahead control action
    xk = A*xk+B*u_nn(k); % implement the state update
    xk_store_nn(:,k) = xk;
end

combined = [xk_store_nn;xk_store_MPC]; % compare the results


%% Plot the results.
my_color = [0.9,0.9,0.9];
close all
f_size = 18; font_size_leg = 15; f_size_leg = 15;

fig1 = figure;
plot(1:K, xk_store_nn(1,:),'-xk','color',[0.2 0.2 0.2],'linewidth',2,'markersize',12); hold on;
plot(1:K, xk_store_MPC(1,:),'+k','color',[0.6 0.6 0.6],'linewidth',2,'markersize',12);
grid on
xlabel('Time instant $k$','interpreter','latex','fontsize',f_size)
ylabel('$x_1[k]$','interpreter','latex','fontsize',f_size)
leg = legend('Implicit NN controller','MPC Controller');
set(leg,'interpreter','latex','fontsize',f_size_leg,'location','southeast')
set(gca,'fontsize',f_size)

fig2 = figure;
plot(1:K, xk_store_nn(2,:),'-xk','color',[0.2 0.2 0.2],'linewidth',2,'markersize',12); hold on;
plot(1:K, xk_store_MPC(2,:),'+k','color',[0.6 0.6 0.6],'linewidth',2,'markersize',12);
grid on
xlabel('Time instant $k$','interpreter','latex','fontsize',f_size)
ylabel('$x_2[k]$','interpreter','latex','fontsize',f_size)
leg = legend('Implicit NN controller','MPC Controller');
set(leg,'interpreter','latex','fontsize',f_size_leg,'location','southeast')
set(gca,'fontsize',f_size)

fig3 = figure;
plot(1:K, u_ramp(1,:),'-xk','color',[0.2 0.2 0.2],'linewidth',2,'markersize',12); hold on;
plot(1:K, u_mpc(:),'+k','color',[0.6 0.6 0.6],'linewidth',2,'markersize',12);
grid on
xlabel('Time instant $k$','interpreter','latex','fontsize',f_size)
ylabel('Input $u[k]$','interpreter','latex','fontsize',f_size)
leg = legend('Implicit NN controller','MPC Controller');
set(leg,'interpreter','latex','fontsize',f_size_leg,'location','southeast')
set(gca,'fontsize',f_size)

fig4 = figure;
set_K = 7;
for j = 1:set_K
plot(1:iters, res_norm_store(j,:),'-k','color',[1 1 1]*j/(set_K+1),'linewidth',2,'markersize',12); hold on;
end
grid on
xlabel('Layer depth $j$','interpreter','latex','fontsize',f_size)
ylabel('Residual $\|w[j]-D\phi(w[j])-\zeta\|_2$','interpreter','latex','fontsize',f_size)
leg = legend('$k = 1$','$k = 2$','$k= 3$','$k= 4$','$k = 5$','$k = 6$', '$k = 7$');
set(leg,'interpreter','latex','fontsize',f_size_leg,'location','northeast')
axis([0 6e2 -0.5 6])
set(gca,'fontsize',f_size)

[X,Y] = meshgrid(1:iters,1:K);
fig5 = figure;
h = surf(X,Y,res_norm_store);
set(h,'edgecolor','none');
% h = contour(X,Y,res_norm_store);

% grid on
xlabel('Layer depth $j$','interpreter','latex','fontsize',f_size)
ylabel('Residual $\|y_j-D\phi_j-\zeta\|_2$','interpreter','latex','fontsize',f_size)
leg = legend('NN controller','MPC Controller');
set(leg,'interpreter','latex','fontsize',f_size_leg,'location','southeast')

%%
% print(fig1,'x1','-depsc');
% print(fig2,'x2','-depsc');
% print(fig3,'u','-depsc');
% print(fig4,'res','-depsc');
% print(fig4,'res','-dpng');






















